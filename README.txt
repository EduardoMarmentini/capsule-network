Dataset de formas geométricas para Capsule Network
====================================================

Imagens em escala de cinza, tamanho 28x28 pixels, fundo preto e forma branca.

Classes:
- circle: círculo
- triangle: triângulo
- square: quadrado

Divisão:
- train/: 500 imagens por classe (1.500 imagens)
- test/: 500 imagens por classe (1.500 imagens)

Total: 3.000 imagens.

As imagens variam em rotação e tamanho. O conjunto de teste pode ser recriado executando `python generate_test_dataset.py`; ele usa uma semente aleatória independente e gera 500 imagens por classe.

Estrutura:
train/circle/
train/triangle/
train/square/
test/circle/
test/triangle/
test/square/

Formato: PNG
