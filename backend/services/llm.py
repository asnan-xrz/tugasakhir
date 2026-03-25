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
Return the output STRICTLY as a JSON object with a single key "scenes" containing an array of objects.
Each object in the array must have exactly these keys: "scene_no" (integer), "location" (string), "description" (string), "shot_type" (string).

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
                    "format": "json",
                    "keep_alive": 0
                }
            )
            response.raise_for_status()
            result: Dict[str, Any] = response.json()
            response_text = result.get("response", "{}")
            
            # Try to parse the JSON response cleanly first
            try:
                scenes = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: Clean up response text in case it contains markdown (like ```json ... ```)
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                    response_text = response_text[start_idx:end_idx+1]
                else:
                    start_idx = response_text.find('[')
                    end_idx = response_text.rfind(']')
                    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                        response_text = response_text[start_idx:end_idx+1]
                
                try:
                    scenes = json.loads(response_text)
                except:
                    scenes = {}
            
            print(f"Ollama returned type: {type(scenes)}")
            try:
                print(f"Parsed JSON keys/length: {str(scenes)[:200]}...")
            except:
                pass
            
            def normalize_scene(raw_scene: Dict[str, Any]) -> Dict[str, Any]:
                norm = {}
                for k, v in raw_scene.items():
                    lk = str(k).lower().replace(' ', '_').replace('-', '_')
                    if 'no' in lk or 'num' in lk or 'id' == lk:
                        norm['scene_no'] = v
                    elif 'loc' in lk:
                        norm['location'] = v
                    elif 'desc' in lk or 'act' in lk or 'scene' == lk:
                        norm['description'] = v
                    elif 'shot' in lk or 'type' in lk:
                        norm['shot_type'] = v
                    else:
                        norm[lk] = v
                return {
                    "scene_no": norm.get('scene_no', 0),
                    "location": norm.get('location', 'Unknown'),
                    "description": norm.get('description', ''),
                    "shot_type": norm.get('shot_type', '')
                }

            extracted_list = []
            if isinstance(scenes, dict):
                # Sometimes Llama returns {"scenes": [...]}
                found_list = False
                for key, val in scenes.items():
                    if isinstance(val, list):
                        extracted_list = val
                        found_list = True
                        break
                if not found_list:
                    extracted_list = [scenes]  # Fallback: single object
            elif isinstance(scenes, list):
                extracted_list = scenes
                
            return [normalize_scene(x) for x in extracted_list if isinstance(x, dict)]
            
    except Exception as e:
        print(f"Error parsing Ollama response: {e}")
        try:
            print(f"Raw response: {response_text}")
        except:
            pass
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("VRAM cache cleared after LLM generation")
        except ImportError:
            pass
        
    return []
