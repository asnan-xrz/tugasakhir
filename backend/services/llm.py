# Stub for LLM generation (e.g. Ollama)

async def enhance_prompt(base_prompt: str) -> str:
    """
    Given a raw prompt, uses a local LLM to expand it into a detailed scene description for Stable Diffusion.
    """
    print(f"Enhancing prompt: {base_prompt}")
    return base_prompt + ", masterpiece, high resolution, detailed, cinematic lighting"
