import torch
import logging
from typing import List, Dict, Any, Generator
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images
from transformers import AutoModelForCausalLM
from src.embedder import EmbedData, ColPaliEmbedder
from src.vector_store import QdrantVectorStore
from src.utils import load_model_with_retry
from src.hpc_colpali import HPC
import numpy as np

logger = logging.getLogger(__name__)

class Retriever:
    """Enhanced retriever with proper HPC handling"""
    
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
        self.hpc = getattr(embedder, 'hpc', None)
    
    def search(self, query: str, limit: int = 4):
        """Search with proper vector handling"""
        try:
            # Get query embedding as flat vector
            query_embedding = self.embedder.embed_query(query)
            
            logger.info(f"Query embedding shape: {len(query_embedding)}")
            logger.info(f"HPC enabled: {self.embedder.use_hpc}")
            logger.info(f"Codebook built: {getattr(self.embedder, 'codebook_built', False)}")
            
            # Search directly with the flat vector
            results = self.vector_store.search(query_embedding, limit=limit)
            
            logger.info(f"Retrieved {len(results.points)} results for query")
            if results.points:
                logger.debug(f"Top result score: {results.points[0].score}")
            
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise

class MultimodalRAG:
    """Production-ready multimodal RAG system"""
    
    def __init__(self,
                 retriever: Retriever,
                 llm_name: str = "deepseek-ai/Janus-Pro-1B",
                 max_new_tokens: int = 4096):
        self.retriever = retriever
        self.llm_name = llm_name
        self.max_new_tokens = max_new_tokens
        self._setup_llm()
    
    def _setup_llm(self):
        """Initialize language model components"""
        try:
            logger.info(f"Loading LLM: {self.llm_name}")
            
            # Load processor
            self.vl_chat_processor = VLChatProcessor.from_pretrained(
                self.llm_name,
                cache_dir="./Janus/hf_cache",
                trust_remote_code=True
            )
            self.tokenizer = self.vl_chat_processor.tokenizer
            
            # Load model
            self.vl_gpt = load_model_with_retry(
                AutoModelForCausalLM,
                self.llm_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir="./Janus/hf_cache"
            ).eval()
            
            logger.info("LLM loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup LLM: {e}")
            raise
    
    def generate_context(self, query: str, top_k: int = 3) -> List[str]:
        """
        Generate context from retrieval results, returning multiple valid image paths.
        """
        try:
            # Retrieve more results to have a better pool for finding valid files
            results = self.retriever.search(query, limit=top_k * 2)
            
            if not results or not results.points:
                logger.warning("No retrieval results found for the query.")
                return []

            valid_context_paths = []
            seen_indices = set()
            
            for result in results.points:
                if len(valid_context_paths) >= top_k:
                    break # Stop once we have enough contexts

                image_index = result.payload.get('image_index')
                if image_index is None:
                    logger.warning(f"Skipping result {result.id} due to missing 'image_index'.")
                    continue

                if image_index in seen_indices:
                    continue # Avoid duplicate pages

                context_path = f"./images/page{image_index}.jpg"
                
                import os
                if os.path.exists(context_path):
                    logger.info(f"Found valid context: {context_path} (Score: {result.score})")
                    valid_context_paths.append(context_path)
                    seen_indices.add(image_index)
                else:
                    logger.warning(f"Context file not found, skipping: {context_path}")

            if not valid_context_paths:
                logger.error("No valid context files could be found for any retrieval results.")
            
            return valid_context_paths
            
        except Exception as e:
            logger.error(f"Context generation failed: {e}", exc_info=True)
            raise
    
    def query(self, query: str) -> Generator[str, None, None]:
        """Generate streaming response with better error handling and multi-image context."""
        try:
            # Get a list of context images
            image_contexts = self.generate_context(query, top_k=3)
            if not image_contexts:
                yield "Sorry, I couldn't find relevant information in the uploaded document for your query. Please make sure your question relates to the content of the PDF."
                return
            
            logger.info(f"Using {len(image_contexts)} context images for the query.")
            
            # Create a placeholder for each image in the prompt
            image_placeholders = "".join(["<image_placeholder>\n" for _ in image_contexts])

            # Enhanced prompt for better context adherence
            qa_prompt = f"""You are an expert AI assistant analyzing several pages from a document. Answer the user's question using ONLY the information visible in the provided images.

**CRITICAL INSTRUCTIONS:**
1.  Base your answer exclusively on the content of the images. Do not use any external knowledge.
2.  Synthesize information from all provided images to form a comprehensive answer.
3.  If the images do not contain the necessary information, state that clearly. For example: "The provided pages do not contain information about [topic]."
4.  Be precise. If you quote text or describe a figure, mention that it's from the provided pages.

**User Question:** {query}

**Your Answer (based only on the provided images):**"""
            
            conversation = [
                {
                    "role": "User",
                    "content": f"{image_placeholders}{qa_prompt}",
                    "images": image_contexts, # Pass the list of image paths
                },
                {"role": "Assistant", "content": ""},
            ]
            
            # Process and generate response
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.vl_chat_processor(
                conversations=conversation, 
                images=pil_images, 
                force_batchify=True
            ).to(self.vl_gpt.device)
            
            # The `vl_gpt` model prepares the final input embeddings that combine text and images.
            inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            # Generate response using the language_model component
            with torch.inference_mode():
                outputs = self.vl_gpt.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            
            # Clean up the response (remove input prompt)
            # The model output includes the conversation history, so we split it off.
            assistant_response_start = response.find("[/INST]")
            if assistant_response_start != -1:
                response = response[assistant_response_start + len("[/INST]"):].strip()

            # Stream the response
            for char in response:
                yield char
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            yield f"Error processing query: {str(e)}. Please try again or re-upload your document."

# Compatibility wrapper
class RAG(MultimodalRAG):
    """Backward compatibility wrapper"""
    
    def __init__(self, retriever, llm_name="deepseek-ai/Janus-Pro-1B"):
        super().__init__(retriever, llm_name)
    
    def query(self, query: str) -> str:
        """Return complete response instead of streaming"""
        response_chars = list(super().query(query))
        return ''.join(response_chars)