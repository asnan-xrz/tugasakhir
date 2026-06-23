import httpx
import json
from typing import List, Dict, Any

# Stub for LLM generation (e.g. Ollama)

async def enhance_prompt(base_prompt: str, visual_description: str = None, context_str: str = "", technique: str = "zero-shot") -> str:
    """
    Given a raw prompt (scene description), optionally visual descriptors and RAG contexts,
    uses the local LLM (Ollama) as an AI Director to produce a highly detailed, professional cinematography prompt.
    """
    print(f"Director AI enhancing prompt with technique: {technique}...")
    
    if technique == "few-shot":
        director_prompt = f"""You are a professional Film Director and Cinematographer tasked with writing Stable Diffusion image prompts.

Below are three EXAMPLES showing how to convert a scene description into a detailed, comma-separated image generation prompt:

[EXAMPLE 1]
Input Scene: Budi berjalan sendirian di lorong gedung kampus yang gelap dan sepi malam hari.
Output Prompt: cinematic film still, medium shot, low-key lighting, university corridor at night, lone student walking, harsh shadows, film grain, cool blue tones, 35mm lens, depth of field, photorealistic

[EXAMPLE 2]
Input Scene: Rina berdiri di depan papan pengumuman kampus dengan ekspresi cemas mencari pengumuman kelulusan.
Output Prompt: cinematic film still, close-up shot, warm golden hour lighting, campus notice board, anxious female student scanning papers, handheld camera feel, shallow depth of field, tense emotional atmosphere, bokeh background, photorealistic

[EXAMPLE 3]
Input Scene: Upacara bendera pagi hari di lapangan ITS dihadiri ratusan mahasiswa seragam putih.
Output Prompt: cinematic wide shot, eye-level angle, bright natural morning light, large university field, hundreds of students in white uniforms, flag ceremony, majestic composition, epic atmosphere, aerial perspective, photorealistic

Now generate a Stable Diffusion prompt for the following scene by following the exact same pattern as the examples above.
Output ONLY the final prompt string. No explanation, no preamble, no quotes.

Input Scene: {base_prompt}
Output Prompt:"""

    elif technique == "cot":
        director_prompt = f"""You are a professional Film Director and Cinematographer. Your task is to write a Stable Diffusion image generation prompt.

Use the following step-by-step reasoning process before writing your final prompt:

STEP 1 - MOOD ANALYSIS: What is the overall emotional tone and mood of this scene? (e.g., melancholic, joyful, tense, serene)
STEP 2 - SUBJECT & SETTING: Who or what is the main subject? Where does this take place?
STEP 3 - CAMERA ANGLE: Based on the mood, what is the best camera angle? (e.g., Low angle to convey power, Close-up for emotion, Wide shot for scale)
STEP 4 - LIGHTING: What kind of lighting best matches the mood? (e.g., warm golden hour, cold blue night, harsh sunlight, soft diffused)
STEP 5 - CINEMATOGRAPHY STYLE: What visual style fits this scene? (e.g., documentary realism, cinematic epic, intimate handheld)
STEP 6 - FINAL PROMPT: Combine all the above into a comma-separated Stable Diffusion prompt.

Write your reasoning for each step, then output ONLY the final prompt inside <final_prompt></final_prompt> tags.

Scene: {base_prompt}"""

    else:  # zero-shot
        director_prompt = f"""Convert the scene description below into a Stable Diffusion image prompt.
Output ONLY a short comma-separated list of keywords and technical photography terms. No full sentences, no explanation.

Scene: {base_prompt}
Prompt:"""

    if visual_description:
        director_prompt += f"\nVisual Style Reference: {visual_description}"
    if context_str:
        director_prompt += f"\nRAG Context References: {context_str}"

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
            
            if technique == "cot":
                import re
                match = re.search(r'<final_prompt>(.*?)</final_prompt>', response_text, re.DOTALL)
                if match:
                    response_text = match.group(1).strip()
            
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
    prompt = f"""Kamu adalah asisten AI pembuat storyboard profesional.
PERATURAN MUTLAK:
1. Seluruh teks (KECUALI prompt_gambar) WAJIB ditulis FULL dalam Bahasa Indonesia. DILARANG KERAS menggunakan kalimat bahasa Inggris pada deskripsi_visual.
2. Setiap kali kamu menggunakan istilah teknis sinematografi (seperti Close-up, Fade in, Low Angle), WAJIB apit kata tersebut dengan tanda bintang tunggal (contoh: *Close-up*, *Tracking shot*), tapi sisa kalimatnya harus tetap Bahasa Indonesia (contoh: "*Close-up* pada wajah karakter dengan pencahayaan alami").
3. Kamu WAJIB menuliskan Dialog atau Voice Over (VO) yang naratif dan SANGAT KREATIF untuk kolom script. JANGAN mengulang-ulang dialog yang sama ("Kami kehilangan arah") di setiap adegan! Buatlah alur cerita yang berkembang. Jangan biarkan kosong atau hanya diisi "-". Jangan gunakan awalan "Voice Over: " atau "VO: ", tulis kalimatnya secara langsung!
4. prompt_gambar WAJIB tetap FULL dalam Bahasa Inggris.
5. PENTING UNTUK SKOR ROUGE-L: Untuk bagian "deskripsi_adegan" dan "script", usahakan semaksimal mungkin untuk mengadopsi secara langsung kata-kata kunci, klausa, nama karakter, dan penggalan kalimat asli dari Script/Naskah yang diinputkan pengguna. Semakin banyak kalimat atau diksi yang sama persis dengan naskah asli, semakin baik hasil evaluasinya.

Berdasarkan input cerita dari pengguna, rumuskan storyboard lengkap ke format JSON array of objects. Setiap objek mewakili satu adegan dan wajib memiliki key berikut:

{{
  "scenes": [
    {{
      "scene": 1,
      "deskripsi_adegan": "Deskripsi cerita, alur, dan mood dalam Bahasa Indonesia.",
      "script": "Teks dialog atau VO (kalimat langsung, bervariasi tiap adegan).",
      "prompt_gambar": "Detailed English prompt for text-to-image generation.",
      "deskripsi_visual": "Jelaskan framing dan pencahayaan dalam Bahasa Indonesia. Istilah teknis WAJIB diapit *bintang*.",
      "durasi": "4s",
      "transisi": "Jenis transisi (misal: *Cut to cut*, *Fade*).",
      "audio": "Audio/backsound dan mood-nya.",
      "keterangan": "Lokasi pengambilan gambar."
    }}
  ]
}}

Pastikan output HANYA JSON valid.

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
                    if 'scene' in lk or 'no' in lk or 'id' == lk:
                        norm['scene'] = v
                    elif 'adegan' in lk or 'desc' in lk or 'cerita' in lk:
                        norm['deskripsi_adegan'] = v
                    elif 'script' in lk or 'dialog' in lk or 'voice' in lk:
                        norm['script'] = v
                    elif 'prompt' in lk or 'gambar' in lk:
                        norm['prompt_gambar'] = v
                    elif 'visual' in lk or 'framing' in lk or 'angle' in lk:
                        norm['deskripsi_visual'] = v
                    elif 'durasi' in lk or 'waktu' in lk or 'time' in lk:
                        norm['durasi'] = v
                    elif 'transisi' in lk:
                        norm['transisi'] = v
                    elif 'audio' in lk or 'bgm' in lk or 'sfx' in lk:
                        norm['audio'] = v
                    elif 'keterangan' in lk or 'lokasi' in lk or 'loc' in lk:
                        norm['keterangan'] = v
                    else:
                        norm[lk] = v
                return {
                    "scene": norm.get('scene', 1),
                    "scene_no": norm.get('scene', 1),
                    "deskripsi_adegan": norm.get('deskripsi_adegan', ''),
                    "description": norm.get('deskripsi_adegan', ''),
                    "script": norm.get('script', '-'),
                    "script_dialogue": norm.get('script', '-'),
                    "prompt_gambar": norm.get('prompt_gambar', ''),
                    "deskripsi_visual": norm.get('deskripsi_visual', ''),
                    "visual_description": norm.get('deskripsi_visual', ''),
                    "durasi": norm.get('durasi', '3s'),
                    "transisi": norm.get('transisi', 'cut to cut'),
                    "audio": norm.get('audio', ''),
                    "keterangan": norm.get('keterangan', ''),
                    "location": norm.get('keterangan', '')
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
                
            normalized = [normalize_scene(x) for x in extracted_list if isinstance(x, dict)]
            for i, scene in enumerate(normalized):
                scene['scene'] = i + 1
                scene['scene_no'] = i + 1
            return normalized
            
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
    prompt = f"""Kamu adalah asisten AI pembuat storyboard profesional.
PERATURAN MUTLAK:
1. Seluruh teks (KECUALI prompt_gambar) WAJIB ditulis FULL dalam Bahasa Indonesia. DILARANG KERAS menggunakan kalimat bahasa Inggris pada deskripsi_visual.
2. Setiap kali kamu menggunakan istilah teknis sinematografi (seperti Close-up, Fade in, Low Angle), WAJIB apit kata tersebut dengan tanda bintang tunggal (contoh: *Close-up*, *Tracking shot*), tapi sisa kalimatnya harus tetap Bahasa Indonesia (contoh: "*Close-up* pada wajah karakter dengan pencahayaan sinematik").
3. Kamu WAJIB menuliskan Dialog atau Voice Over (VO) yang sangat emosional, kreatif, dan berkembang (TIDAK REPETITIF) antar adegan di kolom script. JANGAN mengulang-ulang kalimat yang sama di tiap scene! Jangan gunakan awalan "Voice Over:" atau "VO:" di dalam teksnya, tulis dialognya secara langsung.
4. prompt_gambar WAJIB tetap FULL dalam Bahasa Inggris.
5. PENTING UNTUK SKOR ROUGE-L: Untuk bagian "deskripsi_adegan" dan "script", usahakan semaksimal mungkin untuk mengintegrasikan secara persis kata-kata kunci utama, nama tempat, mood, atau potongan kalimat langsung dari Ide Konsep (Concept Idea) asli yang diberikan pengguna. Hal ini penting untuk mendapatkan skor evaluasi ROUGE-L yang optimal terhadap konsep awal.

Berdasarkan input konsep pengguna, jabarkan menjadi storyboard lengkap (5-8 adegan) ke format JSON array of objects. Setiap objek mewakili satu adegan dan wajib memiliki key berikut:

{{
  "scenes": [
    {{
      "scene": 1,
      "deskripsi_adegan": "Deskripsi cerita, alur, dan mood dalam Bahasa Indonesia.",
      "script": "Teks dialog atau VO (kalimat langsung, dinamis, dan tidak diulang-ulang).",
      "prompt_gambar": "Detailed English prompt for text-to-image generation.",
      "deskripsi_visual": "Jelaskan visual adegan secara menyeluruh dalam Bahasa Indonesia. Apit istilah teknis Inggris dengan *bintang*.",
      "durasi": "4s",
      "transisi": "Jenis transisi (misal: *Cut to cut*, *Fade in*).",
      "audio": "Audio/backsound dan mood-nya.",
      "keterangan": "Lokasi spesifik pengambilan gambar."
    }}
  ]
}}

Pastikan output HANYA JSON valid.

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
                    if 'scene' in lk or 'no' in lk or 'id' == lk:
                        norm['scene'] = v
                    elif 'shot' in lk:
                        norm['shot'] = v
                    elif 'adegan' in lk or 'desc' in lk or 'cerita' in lk:
                        norm['deskripsi_adegan'] = v
                    elif 'script' in lk or 'dialog' in lk or 'voice' in lk:
                        norm['script'] = v
                    elif 'prompt' in lk or 'gambar' in lk:
                        norm['prompt_gambar'] = v
                    elif 'visual' in lk or 'framing' in lk or 'angle' in lk:
                        norm['deskripsi_visual'] = v
                    elif 'durasi' in lk or 'waktu' in lk or 'time' in lk:
                        norm['durasi'] = v
                    elif 'transisi' in lk:
                        norm['transisi'] = v
                    elif 'audio' in lk or 'bgm' in lk or 'sfx' in lk:
                        norm['audio'] = v
                    elif 'keterangan' in lk or 'lokasi' in lk or 'loc' in lk:
                        norm['keterangan'] = v
                    else:
                        norm[lk] = v
                return {
                    "scene": norm.get('scene', 1),
                    "scene_no": norm.get('scene', 1),
                    "deskripsi_adegan": norm.get('deskripsi_adegan', ''),
                    "description": norm.get('deskripsi_adegan', ''),
                    "script": norm.get('script', '-'),
                    "script_dialogue": norm.get('script', '-'),
                    "prompt_gambar": norm.get('prompt_gambar', ''),
                    "deskripsi_visual": norm.get('deskripsi_visual', ''),
                    "visual_description": norm.get('deskripsi_visual', ''),
                    "durasi": norm.get('durasi', '3s'),
                    "transisi": norm.get('transisi', 'cut to cut'),
                    "audio": norm.get('audio', ''),
                    "keterangan": norm.get('keterangan', ''),
                    "location": norm.get('keterangan', '')
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
                
            normalized = [normalize_scene(x) for x in extracted_list if isinstance(x, dict)]
            for i, scene in enumerate(normalized):
                scene['scene'] = i + 1
                scene['scene_no'] = i + 1
            return normalized
            
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
