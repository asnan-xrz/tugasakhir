import httpx
import json
from typing import List, Dict, Any

# Stub for LLM generation (e.g. Ollama)

async def enhance_prompt(base_prompt: str) -> str:
    """
    Given a raw prompt, uses a local LLM to expand it into a detailed scene description for Stable Diffusion.
    """
    print(f"Enhancing prompt: {base_prompt}")
    return base_prompt + ", masterpiece, high resolution, detailed, cinematic lighting"

async def analyze_script_to_scenes(script_text: str) -> List[Dict[str, Any]]:
    """
    Sends the extracted script text to local Ollama and requests a JSON list of scenes.
    """
    prompt = f"""You are a professional storyboard artist and script analyzer.
Analyze the following script text and break it down into a list of scenes.
Return the output STRICTLY as a pure JSON array of objects with exactly these keys: "scene_no" (integer), "location" (string), "description" (string), "shot_type" (string).
Do not include any markdown, explanations, or other text besides the JSON array.

Script:
{script_text}
"""
    
    try:
        from dotenv import load_dotenv
        import os
        
        # Load environment variables from config.env in the root directory
        # Adjust path if config.env is located elsewhere
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.env")
        load_dotenv(config_path)
        
        headers = {}
        api_key = os.getenv("OLLAMA_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                headers=headers,
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                }
            )
            response.raise_for_status()
            result: Dict[str, Any] = response.json()
            response_text = result.get("response", "[]")
            
            # Parse the JSON response
            scenes = json.loads(response_text)
            if isinstance(scenes, list):
                return scenes
    except Exception as e:
        print(f"Error calling Ollama: {e}")
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("VRAM cache cleared after LLM generation")
        except ImportError:
            pass
        
    return []
