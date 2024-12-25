from transformers import MarianMTModel, MarianTokenizer
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import torch

def translate_text(input_text):
    """翻譯中文為英文"""
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # 使用模型進行翻譯
    translated = model.generate(**tokenizer([input_text], return_tensors="pt", padding=True))
    translation = [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
    return translation

def generate_image(description):
    """根據英文描述生成圖片"""
    # 加載 Stable Diffusion 模型
    sd_pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    sd_pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    # 使用模型生成圖片
    image = sd_pipeline(description).images[0]
    return image

def display_image(image, title, translated_text):
    """使用 Matplotlib 顯示圖片，並在上方標註翻譯後的英文"""
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"{title}\n({translated_text})", fontsize=16)
    plt.show()

def main():
    while True:
        # 提示使用者輸入中文描述
        user_input = input("請輸入中文描述（或輸入 'exit' 結束）：")
        if user_input.lower() == "exit":
            print("程式結束。")
            break

        # 步驟 1: 翻譯中文
        print("翻譯中...")
        translated_text = translate_text(user_input)
        print(f"翻譯結果：{translated_text}")

        # 步驟 2: 生成圖片
        print("生成圖片中，請稍候...")
        image = generate_image(translated_text)

        # 步驟 3: 使用 Matplotlib 顯示圖片，並標註翻譯後的英文
        print("顯示圖片...")
        display_image(image, title=user_input, translated_text=translated_text)

        # 儲存圖片
        image.save("generated_image.png")
        print("圖片已儲存為 'generated_image.png'")

if __name__ == "__main__":
    main()
