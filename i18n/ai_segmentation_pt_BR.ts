<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="pt_BR">
<context>
    <name>AISegmentation</name>
    <!-- Welcome Section -->
    <message>
        <source>Click Install to set up AI Segmentation</source>
        <translation>Clique em Instalar para configurar AI Segmentation</translation>
    </message>

    <!-- Setup Section -->
    <message>
        <source>Installing AI Segmentation...</source>
        <translation>Instalando AI Segmentation...</translation>
    </message>
    <message>
        <source>Verifying installation...</source>
        <translation>Verificando instalacao...</translation>
    </message>
    <message>
        <source>Detecting device...</source>
        <translation>Detectando dispositivo...</translation>
    </message>
    <message>
        <source>Install path: {}</source>
        <translation>Caminho de instalacao: {}</translation>
    </message>
    <message>
        <source>To install in a different folder, set the environment variable AI_SEGMENTATION_CACHE_DIR:</source>
        <translation>Para instalar em outra pasta, defina a variavel de ambiente AI_SEGMENTATION_CACHE_DIR:</translation>
    </message>
    <message>
        <source>1. Open Windows Settings &gt; System &gt; Advanced system settings
2. Click 'Environment Variables'
3. Under 'User variables', click 'New'
4. Variable name: AI_SEGMENTATION_CACHE_DIR
5. Variable value: the folder path you want to use
6. Click OK and restart QGIS</source>
        <translation>1. Abrir Configuracoes do Windows &gt; Sistema &gt; Configuracoes avancadas do sistema
2. Clicar em 'Variaveis de Ambiente'
3. Em 'Variaveis do usuario', clicar em 'Novo'
4. Nome da variavel: AI_SEGMENTATION_CACHE_DIR
5. Valor da variavel: o caminho da pasta desejada
6. Clicar OK e reiniciar o QGIS</translation>
    </message>
    <message>
        <source>Run this command in Terminal, then restart QGIS:

launchctl setenv AI_SEGMENTATION_CACHE_DIR /your/path</source>
        <translation>Execute este comando no Terminal e reinicie o QGIS:

launchctl setenv AI_SEGMENTATION_CACHE_DIR /seu/caminho</translation>
    </message>
    <message>
        <source>Add this line to your ~/.bashrc or ~/.profile, then restart QGIS:

export AI_SEGMENTATION_CACHE_DIR=/your/path</source>
        <translation>Adicione esta linha ao seu ~/.bashrc ou ~/.profile e reinicie o QGIS:

export AI_SEGMENTATION_CACHE_DIR=/seu/caminho</translation>
    </message>
    <message>
        <source>Checking...</source>
        <translation>Verificando...</translation>
    </message>
    <message>
        <source>Install</source>
        <translation>Instalar</translation>
    </message>
    <message>
        <source>Update</source>
        <translation>Atualizar</translation>
    </message>
    <message>
        <source>Dependencies ready</source>
        <translation>Dependencias prontas</translation>
    </message>
    <message>
        <source>Downloading AI model...</source>
        <translation>Baixando modelo IA...</translation>
    </message>
    <message>
        <source>Dependencies ready, model not downloaded</source>
        <translation>Dependencias prontas, modelo nao baixado</translation>
    </message>
    <message>
        <source>Dependencies ready, model download failed</source>
        <translation>Dependencias prontas, falha no download do modelo</translation>
    </message>
    <message>
        <source>Download Model</source>
        <translation>Baixar modelo</translation>
    </message>
    <message>
        <source>Cancel</source>
        <translation>Cancelar</translation>
    </message>
    <message>
        <source>Cancel installation</source>
        <translation>Cancelar instalacao</translation>
    </message>
    <message>
        <source>Are you sure you want to cancel the installation?</source>
        <translation>Tem certeza que deseja cancelar a instalacao?</translation>
    </message>
    <message>
        <source>Installation cancelled</source>
        <translation>Instalação cancelada</translation>
    </message>
    <message>
        <source>Installation failed</source>
        <translation>Falha na instalação</translation>
    </message>
    <message>
        <source>Verification failed:</source>
        <translation>Falha na verificação:</translation>
    </message>
    <message>
        <source>Verification Failed</source>
        <translation>Falha na verificação</translation>
    </message>
    <message>
        <source>Virtual environment was created but verification failed:</source>
        <translation>O ambiente virtual foi criado mas a verificação falhou:</translation>
    </message>
    <message>
        <source>Unknown error</source>
        <translation>Erro desconhecido</translation>
    </message>
    <message>
        <source>Installation Failed</source>
        <translation>Falha na instalação</translation>
    </message>

    <!-- Model Section -->
    <message>
        <source>Update QGIS to 3.34+ for the latest AI model</source>
        <translation>Atualize o QGIS para 3.34+ para o modelo de IA mais recente</translation>
    </message>
    <message>
        <source>Intel Mac: using SAM1 (compatible with PyTorch 2.2)</source>
        <translation>Mac Intel: usando SAM1 (compativel com PyTorch 2.2)</translation>
    </message>
    <message>
        <source>Download Failed</source>
        <translation>Falha no download</translation>
    </message>
    <message>
        <source>Failed to download model:</source>
        <translation>Falha ao baixar modelo:</translation>
    </message>

    <!-- Panel Title -->
    <message>
        <source>AI Segmentation by TerraLab</source>
        <translation>AI Segmentation por TerraLab</translation>
    </message>

    <!-- Segmentation Section -->
    <message>
        <source>Select a Raster Layer to Segment:</source>
        <translation>Selecione uma camada raster para segmentar:</translation>
    </message>
    <message>
        <source>Select a raster layer (GeoTIFF, WMS, XYZ tiles, etc.)</source>
        <translation>Selecione uma camada raster (GeoTIFF, WMS, tiles XYZ, etc.)</translation>
    </message>
    <message>
        <source>No raster layer found. Add a GeoTIFF, image file, or online layer (WMS, XYZ) to your project.</source>
        <translation>Nenhuma camada raster encontrada. Adicione um GeoTIFF, arquivo de imagem ou camada online (WMS, XYZ) ao seu projeto.</translation>
    </message>
    <message>
        <source>No layer found. Add a raster or online layer to your project.</source>
        <translation>Nenhuma camada encontrada. Adicione uma camada raster ou online ao seu projeto.</translation>
    </message>
    <message>
        <source>Start AI Segmentation</source>
        <translation>Iniciar AI Segmentation</translation>
    </message>
    <message>
        <source>Save polygon</source>
        <translation>Salvar polígono</translation>
    </message>
    <message>
        <source>Undo last point</source>
        <translation>Desfazer último ponto</translation>
    </message>
    <message>
        <source>Stop segmentation</source>
        <translation>Parar segmentação</translation>
    </message>
    <message>
        <source>Segmentation</source>
        <translation>Segmentação</translation>
    </message>
    <message>
        <source>Navigation</source>
        <translation>Navegação</translation>
    </message>
    <message>
        <source>Space</source>
        <translation>Espaço</translation>
    </message>
    <message>
        <source>Hold and move to pan the map</source>
        <translation>Manter e mover para deslocar o mapa</translation>
    </message>
    <message>
        <source>Middle mouse button</source>
        <translation>Botão central do mouse</translation>
    </message>
    <message>
        <source>Click and drag to pan the map</source>
        <translation>Clicar e arrastar para deslocar o mapa</translation>
    </message>
    <message>
        <source>Shortcuts</source>
        <translation>Atalhos</translation>
    </message>
    <message>
        <source>Save current polygon to your session</source>
        <translation>Salvar poligono atual na sua sessão</translation>
    </message>
    <message>
        <source>The AI model works best on one element at a time.</source>
        <translation>O modelo IA funciona melhor com um elemento de cada vez.</translation>
    </message>
    <message>
        <source>Save your polygon before selecting the next element.</source>
        <translation>Salve seu poligono antes de selecionar o proximo.</translation>
    </message>
    <message>
        <source>Export polygon to a layer</source>
        <translation>Exportar poligono para uma camada</translation>
    </message>
    <message>
        <source>Export {count} polygons to a layer</source>
        <translation>Exportar {count} poligonos para uma camada</translation>
    </message>

    <!-- Refine Section -->
    <message>
        <source>Refine selection</source>
        <translation>Refinar seleção</translation>
    </message>
    <message>
        <source>Expand/Contract:</source>
        <translation>Expandir/Contrair:</translation>
    </message>
    <message>
        <source>Positive = expand outward, Negative = shrink inward</source>
        <translation>Positivo = expandir para fora, Negativo = contrair para dentro</translation>
    </message>
    <message>
        <source>Simplify outline:</source>
        <translation>Simplificar contorno:</translation>
    </message>
    <message>
        <source>Reduce small variations in the outline (0 = no change)</source>
        <translation>Reduzir pequenas variações no contorno (0 = sem alteração)</translation>
    </message>
    <message>
        <source>Fill holes:</source>
        <translation>Preencher buracos:</translation>
    </message>
    <message>
        <source>Fill interior holes in the selection</source>
        <translation>Preencher buracos internos na seleção</translation>
    </message>
    <message>
        <source>Min area:</source>
        <translation>Área mín:</translation>
    </message>
    <message>
        <source>Remove polygons smaller than this area (in pixels)</source>
        <translation>Remover polígonos menores que esta área (em pixels)</translation>
    </message>
    <message>
        <source>Round corners:</source>
        <translation>Arredondar cantos:</translation>
    </message>
    <message>
        <source>Round corners for natural shapes like trees and bushes. Increase 'Simplify outline' for smoother results.</source>
        <translation>Arredondar cantos para formas naturais como árvores e arbustos. Aumente 'Simplificar contorno' para resultados mais suaves.</translation>
    </message>
    <message>
        <source>Outline</source>
        <translation>Contorno</translation>
    </message>
    <message>
        <source>Selection</source>
        <translation>Seleção</translation>
    </message>

    <!-- Instructions -->
    <message>
        <source>Click on the element you want to segment:</source>
        <translation>Clique no elemento que deseja segmentar:</translation>
    </message>
    <message>
        <source>Left-click to select</source>
        <translation>Clique esquerdo para selecionar</translation>
    </message>
    <message>
        <source>Left-click to add more</source>
        <translation>Clique esquerdo para adicionar mais</translation>
    </message>
    <message>
        <source>Right-click to exclude from selection</source>
        <translation>Clique direito para excluir da seleção</translation>
    </message>
    <message>
        <source>Invalid Layer</source>
        <translation>Camada inválida</translation>
    </message>
    <message>
        <source>Layer extent contains invalid coordinates (NaN/Inf). Check the raster file.</source>
        <translation>A extensão da camada contém coordenadas inválidas (NaN/Inf). Verifique o arquivo raster.</translation>
    </message>

    <!-- Dialogs -->
    <message>
        <source>Not Ready</source>
        <translation>Não está pronto</translation>
    </message>
    <message>
        <source>Please wait for the SAM model to load.</source>
        <translation>Por favor, aguarde o carregamento do modelo SAM.</translation>
    </message>
    <message>
        <source>Load Failed</source>
        <translation>Falha no carregamento</translation>
    </message>
    <message>
        <source>Layer Creation Failed</source>
        <translation>Falha na criação da camada</translation>
    </message>
    <message>
        <source>Could not create the output layer.</source>
        <translation>Não foi possível criar a camada de saída.</translation>
    </message>
    <message>
        <source>Export Failed</source>
        <translation>Falha na exportação</translation>
    </message>
    <message>
        <source>Could not save layer to file:</source>
        <translation>Não foi possível salvar a camada no arquivo:</translation>
    </message>
    <message>
        <source>Layer was saved but could not be loaded:</source>
        <translation>A camada foi salva mas não pôde ser carregada:</translation>
    </message>
    <message>
        <source>You have {count} unsaved polygon(s).</source>
        <translation>Você tem {count} polígono(s) não exportado(s).</translation>
    </message>
    <message>
        <source>Changing layer will discard your current segmentation. Continue?</source>
        <translation>Mudar de camada descartará sua segmentação atual. Continuar?</translation>
    </message>
    <message>
        <source>Change Layer?</source>
        <translation>Mudar de camada?</translation>
    </message>
    <message>
        <source>Stop Segmentation?</source>
        <translation>Parar segmentação?</translation>
    </message>
    <message>
        <source>This will discard {count} polygon(s).</source>
        <translation>Isso descartará {count} polígono(s).</translation>
    </message>
    <message>
        <source>Use 'Export to layer' to keep them.</source>
        <translation>Use 'Exportar para camada' para mantê-las.</translation>
    </message>
    <message>
        <source>This will end the current segmentation session.</source>
        <translation>Isso encerrará a sessão de segmentação atual.</translation>
    </message>
    <message>
        <source>Do you want to continue?</source>
        <translation>Deseja continuar?</translation>
    </message>
    <message>
        <source>Edit saved polygon</source>
        <translation>Editar polígono salvo</translation>
    </message>
    <message>
        <source>Warning: you are about to edit an already saved polygon.</source>
        <translation>Atenção: você está prestes a editar um polígono já salvo.</translation>
    </message>
    <message>
        <source>New to AI Segmentation?</source>
        <translation>Novo no AI Segmentation?</translation>
    </message>
    <message>
        <source>Watch our tutorial</source>
        <translation>Assista nosso tutorial</translation>
    </message>

    <!-- About Section -->
    <message>
        <source>Contact us</source>
        <translation>Fale conosco</translation>
    </message>
    <message>
        <source>Bug, question, feature request?</source>
        <translation>Bug, dúvida, sugestão de funcionalidade?</translation>
    </message>
    <message>
        <source>We'd love to hear from you!</source>
        <translation>Adoraríamos ouvir de você!</translation>
    </message>
    <message>
        <source>Copy email address</source>
        <translation>Copiar endereço de email</translation>
    </message>
    <message>
        <source>or</source>
        <translation>ou</translation>
    </message>
    <message>
        <source>Book a video call</source>
        <translation>Agendar uma videochamada</translation>
    </message>
    <message>
        <source>Tutorial</source>
        <translation>Tutorial</translation>
    </message>
    <message>
        <source>Something not working?</source>
        <translation>Algo não está funcionando?</translation>
    </message>
    <message>
        <source>Copy your logs and send them to us, we'll look into it :)</source>
        <translation>Copie seus logs e envie para nós, vamos verificar :)</translation>
    </message>

    <!-- Tooltip -->
    <message>
        <source>Segment elements on raster images using AI</source>
        <translation>Segmentar elementos em imagens raster usando IA</translation>
    </message>

    <!-- Error Report Dialog -->
    <message>
        <source>Copy your logs with the button below and send them to our email.</source>
        <translation>Copie seus logs com o botão abaixo e envie para nosso email.</translation>
    </message>
    <message>
        <source>We'll fix your issue :)</source>
        <translation>Vamos corrigir seu problema :)</translation>
    </message>
    <message>
        <source>1. Click to copy logs</source>
        <translation>1. Clique para copiar logs</translation>
    </message>
    <message>
        <source>2. Click to send to {}</source>
        <translation>2. Clique para enviar para {}</translation>
    </message>
    <message>
        <source>Open email client</source>
        <translation>Abrir cliente de email</translation>
    </message>
    <message>
        <source>Copied!</source>
        <translation>Copiado!</translation>
    </message>

    <!-- SSL / Antivirus error titles -->
    <message>
        <source>SSL Certificate Error</source>
        <translation>Erro de certificado SSL</translation>
    </message>
    <message>
        <source>Installation Blocked</source>
        <translation>Instalação bloqueada</translation>
    </message>

    <message>
        <source>Click is outside the &apos;{layer}&apos; raster. To segment another raster, stop the current segmentation first.</source>
        <translation>O clique está fora do raster '{layer}'. Para segmentar outro raster, pare primeiro a segmentação atual.</translation>
    </message>

    <!-- Update notification -->
    <message>
        <source>Big update dropped — v{version} is here!</source>
        <translation>Grande atualização — a v{version} chegou!</translation>
    </message>
    <message>
        <source>Grab it now</source>
        <translation>Atualize agora</translation>
    </message>

    <!-- Format conversion -->
    <message>
        <source>{ext} format is not directly supported. GDAL is not available.
Please convert your raster to GeoTIFF (.tif) before using AI Segmentation.</source>
        <translation>O formato {ext} não é diretamente suportado. GDAL não está disponível.
Por favor, converta seu raster para GeoTIFF (.tif) antes de usar AI Segmentation.</translation>
    </message>
    <message>
        <source>Cannot open {ext} file. The format may not be supported by your QGIS installation.
Please convert your raster to GeoTIFF (.tif) before using AI Segmentation.</source>
        <translation>Não foi possível abrir o arquivo {ext}. O formato pode não ser suportado pela sua instalação do QGIS.
Por favor, converta seu raster para GeoTIFF (.tif) antes de usar AI Segmentation.</translation>
    </message>
    <message>
        <source>Failed to read {ext} file: {error}
Please convert your raster to GeoTIFF (.tif) manually.</source>
        <translation>Falha ao ler o arquivo {ext}: {error}
Por favor, converta seu raster para GeoTIFF (.tif) manualmente.</translation>
    </message>

    <!-- PyTorch DLL Error Messages -->
    <message>
        <source>PyTorch Error</source>
        <translation>Erro do PyTorch</translation>
    </message>
    <message>
        <source>PyTorch cannot load on Windows</source>
        <translation>PyTorch não pode carregar no Windows</translation>
    </message>
    <message>
        <source>The plugin requires Visual C++ Redistributables to run PyTorch.

Please download and install:
https://aka.ms/vs/17/release/vc_redist.x64.exe

After installation, restart QGIS and try again.</source>
        <translation>O plugin requer Visual C++ Redistributables para executar o PyTorch.

Por favor, baixe e instale:
https://aka.ms/vs/17/release/vc_redist.x64.exe

Após a instalação, reinicie o QGIS e tente novamente.</translation>
    </message>
    <message>
        <source>Prediction Error</source>
        <translation>Erro de predição</translation>
    </message>
    <message>
        <source>Segmentation failed</source>
        <translation>Falha na segmentação</translation>
    </message>
    <message>
        <source>Layer data provider is not available.</source>
        <translation>O provedor de dados da camada nao esta disponivel.</translation>
    </message>
    <message>
        <source>Failed to fetch tiles from the online layer. Check your network connection.</source>
        <translation>Falha ao buscar tiles da camada online. Verifique sua conexao de rede.</translation>
    </message>
    <message>
        <source>Online layer returned blank tiles for this area. Try panning to an area with data coverage.</source>
        <translation>A camada online retornou tiles em branco para esta area. Tente mover o mapa para uma area com cobertura de dados.</translation>
    </message>
    <message>
        <source>Crop Error</source>
        <translation>Erro de recorte</translation>
    </message>
    <message>
        <source>No raster file path available. Please restart segmentation.</source>
        <translation>Nenhum caminho de arquivo raster disponivel. Reinicie a segmentacao.</translation>
    </message>
    <message>
        <source>Encoding Error</source>
        <translation>Erro de codificacao</translation>
    </message>
    <message>
        <source>Report a Bug</source>
        <translation>Reportar um Bug</translation>
    </message>
    <message>
        <source>Disconnected parts detected in your polygon.</source>
        <translation>Partes desconectadas detectadas no seu poligono.</translation>
    </message>
    <message>
        <source>For best accuracy, segment one element at a time.</source>
        <translation>Para melhor precisao, segmente um elemento de cada vez.</translation>
    </message>
    <message>
        <source>Layer: {}</source>
        <translation>Camada: {}</translation>
    </message>
    <message>
        <source>Polygon saved! Click on another element to segment, or export your polygons.</source>
        <translation>Polígono salvo! Clique em outro elemento para segmentar, ou exporte seus polígonos.</translation>
    </message>
    <message>
        <source>Disconnected parts detected. For best accuracy, segment one element at a time.</source>
        <translation>Partes desconectadas detectadas. Para melhor precisão, segmente um elemento por vez.</translation>
    </message>

    <message>
        <source>No element detected at this point. Try clicking on a different area.</source>
        <translation>Nenhum elemento detectado neste ponto. Tente clicar em uma área diferente.</translation>
    </message>

    <message>
        <source>Updating...</source>
        <translation>Atualizando...</translation>
    </message>

    <message>
        <source>Check for Updates</source>
        <translation>Verificar atualizacoes</translation>
    </message>

    <message>
        <source>More from TerraLab...</source>
        <translation>Mais sobre TerraLab...</translation>
    </message>

    <message>
        <source>Missing Visual C++ Redistributable. Install it, restart your computer, then click Retry.</source>
        <translation>Visual C++ Redistributable ausente. Instale-o, reinicie o computador e clique em Tentar novamente.</translation>
    </message>

    <message>
        <source>Retry</source>
        <translation>Tentar novamente</translation>
    </message>
<!-- v1.0.0 strings -->
    <message>
        <source>Cannot Write Export</source>
        <translation>Nao e possivel exportar</translation>
    </message>
    <message>
        <source>Cannot create export directory '{path}': {reason}</source>
        <translation>Nao foi possivel criar o diretorio de exportacao '{path}': {reason}</translation>
    </message>
    <message>
        <source>The export directory '{path}' is not writable. Choose a different location.</source>
        <translation>O diretorio de exportacao '{path}' nao permite escrita. Escolha outro local.</translation>
    </message>
    <message>
        <source>Loading AI model...</source>
        <translation>Carregando o modelo IA...</translation>
    </message>
    <message>
        <source>SAM model ready</source>
        <translation>Modelo SAM pronto</translation>
    </message>
    <message>
        <source>Ready</source>
        <translation>Pronto</translation>
    </message>
    <message>
        <source>Model load failed</source>
        <translation>Falha ao carregar o modelo</translation>
    </message>
    <message>
        <source>Click landed outside the current element — segment one element at a time. Saving the current selection and starting a new one.</source>
        <translation>O clique ficou fora do elemento atual — segmente um elemento de cada vez. A selecao atual e salva e uma nova e iniciada.</translation>
    </message>
    <message>
        <source>New here?</source>
        <translation>Novo por aqui?</translation>
    </message>
    <message>
        <source>Watch the tutorial</source>
        <translation>Ver o tutorial</translation>
    </message>
    <message>
        <source>Network Connection Problem</source>
        <translation>Problema de conexao de rede</translation>
    </message>
    <message>
        <source>Your connection appears unstable or blocked. Check: (1) your internet is working, (2) QGIS > Settings > Options > Network has a proxy configured if you are on a corporate network, (3) your firewall allows connections to pypi.org and files.pythonhosted.org.</source>
        <translation>Sua conexao parece instavel ou bloqueada. Verifique: (1) a Internet esta funcionando, (2) QGIS > Configuracoes > Opcoes > Rede tem um proxy se voce esta em uma rede corporativa, (3) seu firewall permite conexoes a pypi.org e files.pythonhosted.org.</translation>
    </message>
    <message>
        <source>Sign in to TerraLab</source>
        <translation>Entrar no TerraLab</translation>
    </message>
    <message>
        <source>Two steps to start using AI Segmentation</source>
        <translation>Dois passos para começar a usar o AI Segmentation</translation>
    </message>
    <message>
        <source>1. Sign up or sign in on terra-lab.ai to get your key</source>
        <translation>1. Cadastre-se ou faça login em terra-lab.ai para obter sua chave</translation>
    </message>
    <message>
        <source>2. Paste your key below to activate</source>
        <translation>2. Cole sua chave abaixo para ativar</translation>
    </message>
    <message>
        <source>1. Sign up / Sign in</source>
        <translation>1. Cadastre-se / Faça login</translation>
    </message>
    <message>
        <source>Get Your Key</source>
        <translation>Obter sua chave</translation>
    </message>
    <message>
        <source>2. Paste your activation key</source>
        <translation>2. Cole sua chave de ativação</translation>
    </message>
    <message>
        <source>Sign in to get your key</source>
        <translation>Entre para obter sua chave</translation>
    </message>
    <message>
        <source>Create your free TerraLab account or sign in, then copy your activation key from the dashboard.</source>
        <translation>Crie sua conta gratuita TerraLab ou entre, depois copie sua chave de ativacao do painel.</translation>
    </message>
    <message>
        <source>Activate</source>
        <translation>Ativar</translation>
    </message>
    <message>
        <source>Please enter your activation key.</source>
        <translation>Por favor, insira sua chave de ativacao.</translation>
    </message>
    <message>
        <source>Checking...</source>
        <translation>Verificando...</translation>
    </message>
    <message>
        <source>Activation key verified!</source>
        <translation>Chave de ativacao verificada!</translation>
    </message>
    <message>
        <source>Invalid activation key.</source>
        <translation>Chave de ativacao invalida.</translation>
    </message>
    <message>
        <source>Cannot reach server. Check your internet connection.</source>
        <translation>Nao foi possivel conectar ao servidor. Verifique sua conexao com a Internet.</translation>
    </message>
    <message>
        <source>Signed in!</source>
        <translation>Conectado!</translation>
    </message>
    <message>
        <source>AI Edit</source>
        <translation>AI Edit</translation>
    </message>
    <message>
        <source>Generate imagery with AI on map zones (opens AI Edit plugin)</source>
        <translation>Gere imagens de IA em zonas do mapa (abre o plugin AI Edit)</translation>
    </message>
    <message>
        <source>Right-click must be inside the current selection area.</source>
        <translation>O clique direito deve estar dentro da area de selecao atual.</translation>
    </message>
    <!-- Account Settings Dialog -->
    <message>
        <source>Account Settings</source>
        <translation>Configuracoes da conta</translation>
    </message>
    <message>
        <source>Loading account info...</source>
        <translation>Carregando informacoes da conta...</translation>
    </message>
    <message>
        <source>Manage account on terra-lab.ai</source>
        <translation>Gerenciar conta em terra-lab.ai</translation>
    </message>
    <message>
        <source>Show</source>
        <translation>Mostrar</translation>
    </message>
    <message>
        <source>Hide</source>
        <translation>Ocultar</translation>
    </message>
    <message>
        <source>Change activation key</source>
        <translation>Alterar chave de ativacao</translation>
    </message>
    <message>
        <source>Plan</source>
        <translation>Plano</translation>
    </message>
    <message>
        <source>Free</source>
        <translation>Gratuito</translation>
    </message>
    <message>
        <source>Canceled</source>
        <translation>Cancelado</translation>
    </message>
    <message>
        <source>Email</source>
        <translation>Email</translation>
    </message>
    <message>
        <source>Key</source>
        <translation>Chave</translation>
    </message>
</context>
</TS>
