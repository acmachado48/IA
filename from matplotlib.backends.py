from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# Função para converter imagem PNG em PDF
def convert_png_to_pdf(png_path, pdf_path):
    image = Image.open(png_path)
    rgb_image = image.convert('RGB')
    rgb_image.save(pdf_path, "PDF", resolution=100.0)

# Caminhos de entrada (PNGs gerados anteriormente) e saída (PDFs)
png_to_pdf_paths = {
    "fluxograma_levantamento": (
        "/mnt/data/fluxograma_levantamento.png",
        "/mnt/data/fluxograma_levantamento.pdf"
    ),
    "classificacao_trabalhos": (
        "/mnt/data/classificacao_trabalhos.png",
        "/mnt/data/classificacao_trabalhos.pdf"
    )
}

# Converter cada PNG em PDF
for name, (png, pdf) in png_to_pdf_paths.items():
    convert_png_to_pdf(png, pdf)

png_to_pdf_paths
