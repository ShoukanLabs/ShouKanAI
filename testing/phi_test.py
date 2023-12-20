from llama_cpp import Llama
from styletts2 import tts

other_tts = tts.StyleTTS2(model_checkpoint_path='models/epoch_2nd_00013.pth', config_path='models/config_ft.yml')

llm = Llama(model_path="./models/phi-2.Q4_K_M.gguf", n_gpu_layers=30)
output = llm(
    "A peom, by william shakespeare",  # Prompt
    echo=True)  # Echo the prompt back in the output
print(output["choices"][0]["text"])
other_tts.inference(output["choices"][0]["text"], target_voice_path="models/11_14.wav", output_wav_file="models/test.wav", diffusion_steps=50, output_sample_rate=48000)