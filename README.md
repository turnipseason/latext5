# latext5
Проект является идейным продолжением проекта [EMMA](https://github.com/basic-go-ahead/emma).
Репозиторий содержит использовавшиеся для реализации проекта .ipynb файлы, а также примеры использования и результатов нормализации.

### Описание модели:
Модель для нормализации русскоязычных текстов, содержащих математические сущности, в формат LaTeX.<br>
Модель является дообученной на переведённом&аугментированном датасете "[Mathematics Stack Exchange API Q&A Data](https://zenodo.org/records/1414384)" версией модели [cointegrated/rut5-small](https://huggingface.co/cointegrated/rut5-small).
##### [Ссылка на карточку модели на платформе Hugging Faсe](https://huggingface.co/turnipseason/latext5)

### Пример использования модели:
``` python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IPython.display import display, Math, Latex

model_dir = "turnipseason/latext5"
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_latex(text):
  inputs = tokenizer(text, return_tensors='pt').to(device)
  with torch.no_grad():
  hypotheses = model.generate(
  **inputs,
  do_sample=True, num_return_sequences=1,
  repetition_penalty=1.2,
  max_length=len(text),
  num_beams=10,
  early_stopping=True
  )
  for h in hypotheses:
  display(Latex(tokenizer.decode(h, skip_special_tokens=True)))

text = '''лямбда прописная квадрат минус три равно десять игрек куб
        При этом шинус икс равен интеграл от экспоненты до трёх игрек штрих'''
get_latex(text)
```

### Примеры результатов нормализации:
#### Нормализация текстов, содержащих формулы
![long_with_text_LaTeXT5](https://github.com/turnipseason/latext5/assets/100782385/eddde0f4-08f0-49c9-8471-8c9776e29696)
![лямбда_прописная_LaTeXT5](https://github.com/turnipseason/latext5/assets/100782385/bdb9161e-9803-4101-b940-f44ae5d85189)
#### Соответствие разговорным формулировкам русского языка
![sh_phi_LaTeXT5](https://github.com/turnipseason/latext5/assets/100782385/740053b9-a7a5-41cb-bb3c-5282f69aac15)
#### Нормализация длинных текстов и восстановление разметки
![длинная_сумма_LaTeXT5](https://github.com/turnipseason/latext5/assets/100782385/7b1dd21e-ed4f-4034-881f-2b3ebb8b89d3)
#### Нормализация числительных
![числа_в_тексте_LaTeXT5](https://github.com/turnipseason/latext5/assets/100782385/a71ebbe6-3f38-442c-9c0a-0f0d01d46d7e)
