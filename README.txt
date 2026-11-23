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

Classificação de uma imagem no Google Colab
============================================

Monte o Google Drive e deixe `capsule_network.py` acessível no diretório atual
do notebook. Depois, execute:

    from capsule_network import upload_and_predict
    upload_and_predict()

Se executar `capsule_network.py` diretamente no Colab, ele também chama esse
fluxo automaticamente.

Se encontrar `capsule_network_best.pth` compatível, o código carrega o modelo.
Se o checkpoint for antigo ou não existir, treina novamente e salva um novo.
Depois, o Colab abre a janela para selecionar imagens e exibe a classe prevista:
círculo, triângulo ou quadrado. Depois de cada rodada, digite `s` para enviar
mais imagens ou `n` para encerrar. O treino exige as pastas de dados acima no
caminho `DATASET_DIR`. O modelo espera imagens parecidas com as usadas no treino:
fundo preto e forma clara.
