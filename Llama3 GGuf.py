from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Проверка наличия доступного GPU и установка его для использования
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {device}")

# Загрузка модели и токенизатора
model_path = 'D:/work/model/Lexi-Llama-3-8B-Uncensored-GGUF'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Перемещение модели на GPU, если доступен
model.to(device)

# Функция для интерактивного общения с моделью
def chat_with_model():
    print("Введите 'выход' для завершения диалога.")
    while True:
        user_input = input("Вы: ")
        if user_input.lower() == 'выход':
            print("Диалог завершен.")
            break
        input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
        # Генерация ответа модели
        output = model.generate(input_ids, max_length=200, num_return_sequences=1)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Модель: " + decoded_output)

# Запуск функции для общения с моделью
if __name__ == "__main__":
    chat_with_model()
