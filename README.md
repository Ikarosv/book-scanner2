# Book Scanner Antigravity

Um utilitário de linha de comando robusto e inteligente em Python para processar fotos cruas de páginas de livros antigos e convertê-las em PDFs limpos, centralizados e opcionalmente pesquisáveis (com OCR).

Diferente de scanners de mesa, tirar fotos de livros com câmeras ou celulares introduz diversos artefatos físicos difíceis: fundos pretos da mesa, dedos segurando as bordas, páginas que rotacionam sozinhas nos sensores do celular, páginas duplas distorcidas, inclinação do papel, e iluminação sombreada da lombada.

Este projeto resolve todos estes problemas usando matemática avançada e IA para gerar um PDF puro e reto, simulando o resultado de um scanner industrial.

## 🌟 Funcionalidades Principais

1. **Recorte Geométrico Inteligente (Cropping):** 
   Localiza a folha de papel claro contra fundos escuros (mesmo com mãos segurando), recorta e achata a imagem original expulsando todo o cenário externo.
2. **Separação Dinâmica de Páginas Duplas:**
   Lida perfeitamente com fotos de livros abertos. Localiza a sombra exata da lombada usando um filtro de tendência estatística e divide a foto precisamente de cima a baixo nas "Página A" e "Página B" originais.
3. **Deskewing de Precisão (Nivelamento Fino):**
   Humanos raramente tiram fotos 100% paralelas. O script agrupa os pixels das palavras, encontra a mediana matemática de inclinação do parágrafo escrito e altera fisicamente a rotação da imagem (-15º a +15º) até que as linhas de texto fiquem estritamente na horizontal.
4. **Votação OCR Universal para Orientação Final:**
   Câmeras erram o EXIF e páginas quadradas enganam IAs básicas. O script aplica OCR nas 4 direções possíveis simultaneamente e escolhe matematicamente o sentido com maior número de palavras legíveis. Fim das fotos de cabeça para baixo ou deitadas para o lado!
5. **Limpeza e Binarização Avançada:**
   Elimina sombras das laterais grossas dividindo o quadro pelo fundo da imagem. Foca e escurece os textos em preto puro enquanto as páginas mais antigas e encardidas viram o mais claro branco digital.
6. **Geração Dupla de PDFs:**
   Produz um bloco mestre em PDF das imagens sem tratamento de texto, e opcionalmente `--ocr`, uma versão que injeta os textos sobrepostos da imagem para uso de (Ctrl+F).

## 🧰 Requisitos de Sistema

- **Python 3.8+**
- Instalação no SO do **Tesseract-OCR** (Certifique-se de que o comando `tesseract` esteja disponível no seu `PATH`, ou atualize o caminho dele na cabeceria do script `book_scanner.py`).

## 📥 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/Ikarosv/book-scanner2.git
cd book-scanner-antigravity
```

2. Instale as dependências Python necessárias:
```bash
pip install -r requirements.txt
```

## 🚀 Como Usar

Estruture seu diretório de trabalho com suas fotos cruas numa pasta separada.

1. Coloque todas as suas fotos (`.jpg`, `.png`, etc.) na pasta `./input`.
2. Rode o comando central indicando sua fonte e destino (opcionalmente passando a flag OCR):

```bash
python book_scanner.py ./input ./output --ocr
```

3. A mágica acontecerá no seu terminal página por página.
4. Acesse a pasta `/output` e recolha seus resquícios finais:
   - `page_XXXX.png` (Todas as páginas finais limpas e individuas para reuso livre).
   - `output_no_ocr.pdf` (Um e-book digital contínuo de altíssima qualidade visual).
   - `output_ocr.pdf` (A versão do leitor com IA caso o comando `--ocr` tenha sido invocado).

---
*Construído para resgatar os livros perdidos nas gavetas.*
