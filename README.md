# VisionCheck
Projeto escolar: classificação experimental de imagens de caneta com tampa, sem tampa e fundo sem caneta.

## Abrir
Site estático, sem banco, senha ou servidor de IA. Publique o repositório em hospedagem HTTPS (Vercel, por exemplo).
A página principal é index.html. index-1.html é a demonstração original simulada, mantida como referência.
Os scripts de TensorFlow.js 4.22.0, MobileNet 2.1.1 e os pesos do modelo são baixados da internet. Não funciona offline na primeira carga.

## Testar no celular
1. Abra em Chrome/Android ou Safari/iOS atualizado, por HTTPS.
2. Abra a câmera e permita acesso. Se não funcionar, envie imagens.
3. Selecione a classe correta e capture ao menos 3 imagens diferentes por classe (recomendado 20 ou mais, variando luz, enquadramento e fundo).
4. A tampa precisa cobrir a ponta. Tampa ao lado ou no fundo da caneta conta como sem tampa.
5. Analise imagens NOVAS, fora do treinamento; registre erros.
6. Exporte o treinamento em JSON para conservar as características numéricas; importe depois ou em outro celular.

## Limites
MobileNet extrai características; um classificador de cinco vizinhos compara com as amostras do usuário. Não há treinamento de detector/localizador da tampa.
O limiar mínimo de exemplos só libera o experimento; não representa validação de precisão.
A concordância entre vizinhos não é probabilidade de acerto.
Imagens semelhantes ao fundo podem enganar o modelo. Resultado inconclusivo não significa que a peça falta.
As duas fotos de referência não são amostras automáticas. Não foi fornecido nem validado um modelo de tampa pronto.
Câmera não é garantida em todo aparelho. Há fallback de arquivo; fotos muito grandes podem precisar de redução.
A inferência ocorre localmente; não enviamos imagens para uma API. Downloads do modelo e bibliotecas exigem rede.
Amostras e histórico ficam só na memória da aba. Exporte antes de fechar. Até 60 amostras por classe e 50 registros da sessão.

## Arquivos
- index.html: interface
- style.css: layout responsivo
- app.js: câmera, uploads, MobileNet e interface
- classifier-core.js: dados, validação e classificação
- tests/classifier.test.cjs: testes matemáticos (não substituem ensaio com imagens reais)

## Validação
node --check app.js
node --check classifier-core.js
node --test tests/*.test.cjs

Antes da apresentação, testar em celular real: aceitar/negar permissão, trocar/desligar câmera, enviar JPG/PNG, treinar classes, exportar/importar e testar objetos fora do treino.
