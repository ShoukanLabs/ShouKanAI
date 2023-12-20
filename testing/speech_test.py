from styletts2 import tts

other_tts = tts.StyleTTS2(model_checkpoint_path='models/epoch_2nd_00013.pth', config_path='models/config_ft.yml')

# Specify target voice to clone. When no target voice is provided, a default voice will be used.
other_tts.inference("this took around 0.9 seconds to generate in under 2 gigabytes of video memory!", target_voice_path="models/11_14.wav", output_wav_file="models/test.wav", diffusion_steps=50, output_sample_rate=48000)