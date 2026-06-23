import asyncio
from backend.services.diffusion import generate_image
async def main():
    try:
        await generate_image("Test prompt", "test.png")
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
