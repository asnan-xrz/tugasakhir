import httpx
import json
from typing import List, Dict, Any

# Stub for LLM generation (e.g. Ollama)

async def enhance_prompt(base_prompt: str, visual_description: str = None, context_str: str = "") -> str:
    """
    Given a raw prompt (scene description), optionally visual descriptors and RAG contexts,
    uses the local LLM (Ollama) as an AI Director to produce a highly detailed, professional cinematography prompt.
    """
    print(f"Director AI enhancing prompt...")
    
    director_prompt = f"""You are a professional Film Director and Cinematographer.
Your job is to translate the following scene description into a highly technical, comma-separated image generation prompt for Stable Diffusion.
Focus on extracting the location, action, characters, and applying professional cinematography terms: Camera Angle (e.g., Low angle, High angle, Eye level), Framing (e.g., Close Up, Medium Shot, Long Shot), and Lighting (e.g., Warm, Cinematic, Natural).
Return ONLY the final prompt string without any conversational text or quotes.

Scene Context: {base_prompt}
"""
    if visual_description:
        director_prompt += f"\nTarget Visual Style (Director's Vision): {visual_description}"
    if context_str:
        director_prompt += f"\nHistorical Visual References to integrate: {context_str}"
        
    director_prompt += "\n\nFormat: masterpiece, high resolution, 8k, highly detailed, [cinematography terms], [location details], [action details], cinematic lighting"

    try:
        from dotenv import load_dotenv
        import os
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
                    "prompt": director_prompt,
                    "stream": False,
                    "keep_alive": 0
                }
            )
            response.raise_for_status()
            result: Dict[str, Any] = response.json()
            response_text = result.get("response", "").strip().strip('"').strip("'")
            
            # Additional safety net just in case Llama outputs conversational junk like "Here is the prompt: "
            prefixes_to_strip = ["Here is", "The prompt", "Here's", "Sure"]
            for prefix in prefixes_to_strip:
                if response_text.startswith(prefix):
                    parts = response_text.split(":", 1)
                    if len(parts) > 1:
                        response_text = parts[1].strip()
                        break
                        
            return response_text
    except Exception as e:
        print(f"Director AI LLM failed: {e}. Falling back to default appending.")
        fallback = f"{base_prompt}, masterpiece, high resolution, detailed, cinematic lighting"
        if visual_description:
            fallback += f", {visual_description}"
        return fallback
    finally:
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("VRAM cache cleared after Director AI execution")
        except ImportError:
            pass

async def analyze_script_to_scenes(script_text: str) -> List[Dict[str, Any]]:
    """
    Sends the extracted script text to local Ollama and requests a JSON list of scenes.
    """
    prompt = f"""You are a strict JSON data extractor.
Read the script text below and extract a list of scenes.
Respond ONLY with a raw JSON object and nothing else. No conversational text, no markdown code blocks, no preamble, and no explanation.
Your output MUST perfectly match this exact JSON structure:
{{
  "scenes": [
    {{
      "scene_no": 1,
      "location": "Name of the location",
      "description": "Detailed description of what happens in the scene",
      "shot_type": "The camera shot type",
      "visual_description": "Cinematography details: Camera Angle (Low/High/Eye level), Framing (CU/MS/LS), and Lighting (Warm/Cinematic/Natural) assigned based on context",
      "script_dialogue": "Dialog or Voice Over (VO) spoken by talent in this scene, or empty string if no voice is present"
    }}
  ]
}}

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

        async with httpx.AsyncClient(timeout=300.0) as client:
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
                import re
                # Pattern match to extract JSON array if preamble or trailing text exists
                array_match = re.search(r'\[[\s\S]*\]', response_text)
                if array_match:
                    try:
                        scenes = json.loads(array_match.group(0))
                    except:
                        pass
                
                # If still not successfully loaded into a dict/list
                if not isinstance(scenes, (dict, list)):
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}')
                    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                        cleaned_text = response_text[start_idx:end_idx+1]
                        try:
                            scenes = json.loads(cleaned_text)
                        except:
                            scenes = {}
                    else:
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
                    elif 'visual' in lk or 'cinematography' in lk:
                        norm['visual_description'] = v
                    elif 'dialog' in lk or 'script' in lk or 'vo' == lk or 'voice' in lk:
                        norm['script_dialogue'] = v
                    else:
                        norm[lk] = v
                return {
                    "scene_no": norm.get('scene_no', 0),
                    "location": norm.get('location', 'Unknown'),
                    "description": norm.get('description', ''),
                    "shot_type": norm.get('shot_type', ''),
                    "visual_description": norm.get('visual_description', ''),
                    "script_dialogue": norm.get('script_dialogue', '')
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
        print(f"Error parsing Ollama response: {type(e).__name__} - {str(e)}")
        try:
            print(f"Raw response: {response_text}")
        except:
            print("No raw response obtained (Possible Timeout/Connection Error)")
    finally:
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("VRAM cache cleared after LLM generation")
        except ImportError:
            pass
        
    return []

async def generate_scenes_from_concept(concept: str) -> List[Dict[str, Any]]:
    """
    Takes a short creative concept and utilizes the local Ollama LLM as a scriptwriter and storyboarder 
    to automatically generate a full 5-8 scene script breakdown.
    """
    prompt = f"""You are a professional Creative Director, Scriptwriter, and Storyboard Artist.
The user has provided a short concept for a video.
Your task is to expand this concept into a full, engaging storyboard consisting of 5 to 8 well-paced scenes.
Respond ONLY with a raw JSON object and nothing else. No conversational text, no markdown code blocks, no preamble.
Your output MUST perfectly match this exact JSON structure:
{{
  "scenes": [
    {{
      "scene_no": 1,
      "location": "Name of the detailed location",
      "description": "Detailed description of what happens visually and narratively",
      "shot_type": "The camera shot type (e.g. Medium Shot, Tracking)",
      "visual_description": "Cinematography details: Camera Angle, Framing, and Lighting",
      "script_dialogue": "Dialog or Voice Over (VO) spoken by talent in this scene, or empty string if no voice is present"
    }}
  ]
}}

Concept Idea:
{concept}
"""
    
    try:
        from dotenv import load_dotenv
        import os
        
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.env")
        load_dotenv(config_path)
        
        headers = {}
        api_key = os.getenv("OLLAMA_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=300.0) as client:
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
            
            try:
                scenes = json.loads(response_text)
            except json.JSONDecodeError:
                import re
                array_match = re.search(r'\[[\s\S]*\]', response_text)
                if array_match:
                    try:
                        scenes = json.loads(array_match.group(0))
                    except:
                        pass
                
                if not isinstance(scenes, (dict, list)):
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}')
                    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                        cleaned_text = response_text[start_idx:end_idx+1]
                        try:
                            scenes = json.loads(cleaned_text)
                        except:
                            scenes = {}
                    else:
                        scenes = {}
            
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
                    elif 'visual' in lk or 'cinematography' in lk:
                        norm['visual_description'] = v
                    elif 'dialog' in lk or 'script' in lk or 'vo' == lk or 'voice' in lk:
                        norm['script_dialogue'] = v
                    else:
                        norm[lk] = v
                return {
                    "scene_no": norm.get('scene_no', 0),
                    "location": norm.get('location', 'Unknown'),
                    "description": norm.get('description', ''),
                    "shot_type": norm.get('shot_type', ''),
                    "visual_description": norm.get('visual_description', ''),
                    "script_dialogue": norm.get('script_dialogue', '')
                }

            extracted_list = []
            if isinstance(scenes, dict):
                found_list = False
                for key, val in scenes.items():
                    if isinstance(val, list):
                        extracted_list = val
                        found_list = True
                        break
                if not found_list:
                    extracted_list = [scenes]
            elif isinstance(scenes, list):
                extracted_list = scenes
                
            return [normalize_scene(x) for x in extracted_list if isinstance(x, dict)]
            
    except Exception as e:
        print(f"Error generating full scenes: {type(e).__name__} - {str(e)}")
    finally:
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("VRAM cache cleared after Concept Extrapolation")
        except ImportError:
            pass
        
    return []
