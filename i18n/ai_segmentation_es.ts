<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="es">
<context>
    <name>AISegmentation</name>
    <!-- Welcome Section -->
    <message>
        <source>Click Install to set up AI Segmentation</source>
        <translation>Haga clic en Instalar para configurar AI Segmentation</translation>
    </message>

    <!-- Setup Section -->
    <message>
        <source>Installing AI Segmentation...</source>
        <translation>Instalando AI Segmentation...</translation>
    </message>
    <message>
        <source>Verifying installation...</source>
        <translation>Verificando instalacion...</translation>
    </message>
    <message>
        <source>Detecting device...</source>
        <translation>Detectando dispositivo...</translation>
    </message>
    <message>
        <source>Install path: {}</source>
        <translation>Ruta de instalacion: {}</translation>
    </message>
    <message>
        <source>To install in a different folder, set the environment variable AI_SEGMENTATION_CACHE_DIR:</source>
        <translation>Para instalar en otra carpeta, defina la variable de entorno AI_SEGMENTATION_CACHE_DIR:</translation>
    </message>
    <message>
        <source>1. Open Windows Settings &gt; System &gt; Advanced system settings
2. Click 'Environment Variables'
3. Under 'User variables', click 'New'
4. Variable name: AI_SEGMENTATION_CACHE_DIR
5. Variable value: the folder path you want to use
6. Click OK and restart QGIS</source>
        <translation>1. Abrir Configuracion de Windows &gt; Sistema &gt; Configuracion avanzada del sistema
2. Hacer clic en 'Variables de entorno'
3. En 'Variables de usuario', hacer clic en 'Nueva'
4. Nombre de la variable: AI_SEGMENTATION_CACHE_DIR
5. Valor de la variable: la ruta de la carpeta deseada
6. Hacer clic en Aceptar y reiniciar QGIS</translation>
    </message>
    <message>
        <source>Run this command in Terminal, then restart QGIS:

launchctl setenv AI_SEGMENTATION_CACHE_DIR /your/path</source>
        <translation>Ejecute este comando en Terminal y reinicie QGIS:

launchctl setenv AI_SEGMENTATION_CACHE_DIR /su/ruta</translation>
    </message>
    <message>
        <source>Add this line to your ~/.bashrc or ~/.profile, then restart QGIS:

export AI_SEGMENTATION_CACHE_DIR=/your/path</source>
        <translation>Agregue esta linea a su ~/.bashrc o ~/.profile y reinicie QGIS:

export AI_SEGMENTATION_CACHE_DIR=/su/ruta</translation>
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
        <translation>Actualizar</translation>
    </message>
    <message>
        <source>Dependencies ready</source>
        <translation>Dependencias listas</translation>
    </message>
    <message>
        <source>Downloading AI model...</source>
        <translation>Descargando modelo IA...</translation>
    </message>
    <message>
        <source>Dependencies ready, model not downloaded</source>
        <translation>Dependencias listas, modelo no descargado</translation>
    </message>
    <message>
        <source>Dependencies ready, model download failed</source>
        <translation>Dependencias listas, fallo en la descarga del modelo</translation>
    </message>
    <message>
        <source>Download Model</source>
        <translation>Descargar modelo</translation>
    </message>
    <message>
        <source>Cancel</source>
        <translation>Cancelar</translation>
    </message>
    <message>
        <source>Cancel installation</source>
        <translation>Cancelar instalacion</translation>
    </message>
    <message>
        <source>Are you sure you want to cancel the installation?</source>
        <translation>Esta seguro de que desea cancelar la instalacion?</translation>
    </message>
    <message>
        <source>Installation cancelled</source>
        <translation>Instalación cancelada</translation>
    </message>
    <message>
        <source>Installation failed</source>
        <translation>Error en la instalación</translation>
    </message>
    <message>
        <source>Verification failed:</source>
        <translation>Error en la verificación:</translation>
    </message>
    <message>
        <source>Verification Failed</source>
        <translation>Error en la verificación</translation>
    </message>
    <message>
        <source>Virtual environment was created but verification failed:</source>
        <translation>El entorno virtual se creó pero la verificación falló:</translation>
    </message>
    <message>
        <source>Unknown error</source>
        <translation>Error desconocido</translation>
    </message>
    <message>
        <source>Installation Failed</source>
        <translation>Error en la instalación</translation>
    </message>

    <!-- Model Section -->
    <message>
        <source>Update QGIS to 3.34+ for the latest AI model</source>
        <translation>Actualice QGIS a 3.34+ para el ultimo modelo de IA</translation>
    </message>
    <message>
        <source>Intel Mac: using SAM1 (compatible with PyTorch 2.2)</source>
        <translation>Mac Intel: usando SAM1 (compatible con PyTorch 2.2)</translation>
    </message>
    <message>
        <source>Download Failed</source>
        <translation>Error en la descarga</translation>
    </message>
    <message>
        <source>Failed to download model:</source>
        <translation>Error al descargar modelo:</translation>
    </message>

    <!-- Panel Title -->
    <message>
        <source>AI Segmentation by TerraLab</source>
        <translation>AI Segmentation por TerraLab</translation>
    </message>

    <!-- Segmentation Section -->
    <message>
        <source>Select a Raster Layer to Segment:</source>
        <translation>Seleccione una capa ráster para segmentar:</translation>
    </message>
    <message>
        <source>Select a raster layer (GeoTIFF, WMS, XYZ tiles, etc.)</source>
        <translation>Seleccione una capa raster (GeoTIFF, WMS, tiles XYZ, etc.)</translation>
    </message>
    <message>
        <source>No raster layer found. Add a GeoTIFF, image file, or online layer (WMS, XYZ) to your project.</source>
        <translation>No se encontro capa raster. Agregue un GeoTIFF, archivo de imagen o capa en linea (WMS, XYZ) a su proyecto.</translation>
    </message>
    <message>
        <source>No layer found. Add a raster or online layer to your project.</source>
        <translation>No se encontro capa. Agregue una capa raster o en linea a su proyecto.</translation>
    </message>
    <message>
        <source>Start AI Segmentation</source>
        <translation>Iniciar AI Segmentation</translation>
    </message>
    <message>
        <source>Save polygon</source>
        <translation>Guardar polígono</translation>
    </message>
    <message>
        <source>Undo last point</source>
        <translation>Deshacer último punto</translation>
    </message>
    <message>
        <source>Stop segmentation</source>
        <translation>Detener segmentación</translation>
    </message>
    <message>
        <source>Segmentation</source>
        <translation>Segmentación</translation>
    </message>
    <message>
        <source>Navigation</source>
        <translation>Navegación</translation>
    </message>
    <message>
        <source>Space</source>
        <translation>Espacio</translation>
    </message>
    <message>
        <source>Hold and move to pan the map</source>
        <translation>Mantener y mover para desplazar el mapa</translation>
    </message>
    <message>
        <source>Middle mouse button</source>
        <translation>Botón central del ratón</translation>
    </message>
    <message>
        <source>Click and drag to pan the map</source>
        <translation>Clic y arrastrar para desplazar el mapa</translation>
    </message>
    <message>
        <source>Shortcuts</source>
        <translation>Atajos</translation>
    </message>
    <message>
        <source>Save current polygon to your session</source>
        <translation>Guardar poligono actual en su sesión</translation>
    </message>
    <message>
        <source>The AI model works best on one element at a time.</source>
        <translation>El modelo IA funciona mejor con un elemento a la vez.</translation>
    </message>
    <message>
        <source>Save your polygon before selecting the next element.</source>
        <translation>Guarda tu poligono antes de seleccionar el siguiente.</translation>
    </message>
    <message>
        <source>Export polygon to a layer</source>
        <translation>Exportar poligono a una capa</translation>
    </message>
    <message>
        <source>Export {count} polygons to a layer</source>
        <translation>Exportar {count} poligonos a una capa</translation>
    </message>

    <!-- Refine Section -->
    <message>
        <source>Refine selection</source>
        <translation>Refinar selección</translation>
    </message>
    <message>
        <source>Expand/Contract:</source>
        <translation>Expandir/Contraer:</translation>
    </message>
    <message>
        <source>Positive = expand outward, Negative = shrink inward</source>
        <translation>Positivo = expandir hacia afuera, Negativo = contraer hacia adentro</translation>
    </message>
    <message>
        <source>Simplify outline:</source>
        <translation>Simplificar contorno:</translation>
    </message>
    <message>
        <source>Reduce small variations in the outline (0 = no change)</source>
        <translation>Reducir pequeñas variaciones en el contorno (0 = sin cambios)</translation>
    </message>
    <message>
        <source>Fill holes:</source>
        <translation>Rellenar huecos:</translation>
    </message>
    <message>
        <source>Fill interior holes in the selection</source>
        <translation>Rellenar huecos interiores en la selección</translation>
    </message>
    <message>
        <source>Min area:</source>
        <translation>Área mín:</translation>
    </message>
    <message>
        <source>Remove polygons smaller than this area (in pixels)</source>
        <translation>Eliminar polígonos menores que esta área (en píxeles)</translation>
    </message>
    <message>
        <source>Round corners:</source>
        <translation>Redondear esquinas:</translation>
    </message>
    <message>
        <source>Round corners for natural shapes like trees and bushes. Increase 'Simplify outline' for smoother results.</source>
        <translation>Redondear esquinas para formas naturales como árboles y arbustos. Aumenta 'Simplificar contorno' para resultados más suaves.</translation>
    </message>
    <message>
        <source>Outline</source>
        <translation>Contorno</translation>
    </message>
    <message>
        <source>Selection</source>
        <translation>Selección</translation>
    </message>

    <!-- Instructions -->
    <message>
        <source>Click on the element you want to segment:</source>
        <translation>Haga clic en el elemento que desea segmentar:</translation>
    </message>
    <message>
        <source>Left-click to select</source>
        <translation>Clic izquierdo para seleccionar</translation>
    </message>
    <message>
        <source>Left-click to add more</source>
        <translation>Clic izquierdo para agregar más</translation>
    </message>
    <message>
        <source>Right-click to exclude from selection</source>
        <translation>Clic derecho para excluir de la selección</translation>
    </message>
    <message>
        <source>Invalid Layer</source>
        <translation>Capa inválida</translation>
    </message>
    <message>
        <source>Layer extent contains invalid coordinates (NaN/Inf). Check the raster file.</source>
        <translation>La extensión de la capa contiene coordenadas inválidas (NaN/Inf). Verifique el archivo ráster.</translation>
    </message>

    <!-- Dialogs -->
    <message>
        <source>Not Ready</source>
        <translation>No está listo</translation>
    </message>
    <message>
        <source>Please wait for the SAM model to load.</source>
        <translation>Por favor, espere a que se cargue el modelo SAM.</translation>
    </message>
    <message>
        <source>Load Failed</source>
        <translation>Error al cargar</translation>
    </message>
    <message>
        <source>Layer Creation Failed</source>
        <translation>Error al crear la capa</translation>
    </message>
    <message>
        <source>Could not create the output layer.</source>
        <translation>No se pudo crear la capa de salida.</translation>
    </message>
    <message>
        <source>Export Failed</source>
        <translation>Error en la exportación</translation>
    </message>
    <message>
        <source>Could not save layer to file:</source>
        <translation>No se pudo guardar la capa en el archivo:</translation>
    </message>
    <message>
        <source>Layer was saved but could not be loaded:</source>
        <translation>La capa se guardó pero no se pudo cargar:</translation>
    </message>
    <message>
        <source>You have {count} unsaved polygon(s).</source>
        <translation>Tiene {count} polígono(s) no exportado(s).</translation>
    </message>
    <message>
        <source>Changing layer will discard your current segmentation. Continue?</source>
        <translation>Cambiar de capa descartará tu segmentación actual. ¿Continuar?</translation>
    </message>
    <message>
        <source>Change Layer?</source>
        <translation>¿Cambiar de capa?</translation>
    </message>
    <message>
        <source>Stop Segmentation?</source>
        <translation>¿Detener segmentación?</translation>
    </message>
    <message>
        <source>This will discard {count} polygon(s).</source>
        <translation>Esto descartará {count} polígono(s).</translation>
    </message>
    <message>
        <source>Use 'Export to layer' to keep them.</source>
        <translation>Use 'Exportar a capa' para conservarlas.</translation>
    </message>
    <message>
        <source>This will end the current segmentation session.</source>
        <translation>Esto finalizará la sesión de segmentación actual.</translation>
    </message>
    <message>
        <source>Do you want to continue?</source>
        <translation>¿Desea continuar?</translation>
    </message>
    <message>
        <source>Edit saved polygon</source>
        <translation>Editar polígono guardado</translation>
    </message>
    <message>
        <source>Warning: you are about to edit an already saved polygon.</source>
        <translation>Advertencia: está a punto de editar un polígono ya guardado.</translation>
    </message>
    <message>
        <source>New to AI Segmentation?</source>
        <translation>¿Nuevo en AI Segmentation?</translation>
    </message>
    <message>
        <source>Watch our tutorial</source>
        <translation>Vea nuestro tutorial</translation>
    </message>

    <!-- About Section -->
    <message>
        <source>Contact us</source>
        <translation>Contáctanos</translation>
    </message>
    <message>
        <source>Bug, question, feature request?</source>
        <translation>¿Bug, pregunta, solicitud de función?</translation>
    </message>
    <message>
        <source>We'd love to hear from you!</source>
        <translation>¡Nos encantaría saber de ti!</translation>
    </message>
    <message>
        <source>Copy email address</source>
        <translation>Copiar dirección de email</translation>
    </message>
    <message>
        <source>or</source>
        <translation>o</translation>
    </message>
    <message>
        <source>Book a video call</source>
        <translation>Reservar una videollamada</translation>
    </message>
    <message>
        <source>Tutorial</source>
        <translation>Tutorial</translation>
    </message>
    <message>
        <source>Something not working?</source>
        <translation>¿Algo no funciona?</translation>
    </message>
    <message>
        <source>Copy your logs and send them to us, we'll look into it :)</source>
        <translation>Copia tus logs y envíanoslos, lo revisaremos :)</translation>
    </message>

    <!-- Tooltip -->
    <message>
        <source>Segment elements on raster images using AI</source>
        <translation>Segmentar elementos en imagenes raster usando IA</translation>
    </message>

    <!-- Error Report Dialog -->
    <message>
        <source>Copy your logs with the button below and send them to our email.</source>
        <translation>Copia tus logs con el botón de abajo y envíalos a nuestro email.</translation>
    </message>
    <message>
        <source>We'll fix your issue :)</source>
        <translation>Solucionaremos tu problema :)</translation>
    </message>
    <message>
        <source>1. Click to copy logs</source>
        <translation>1. Clic para copiar logs</translation>
    </message>
    <message>
        <source>2. Click to send to {}</source>
        <translation>2. Clic para enviar a {}</translation>
    </message>
    <message>
        <source>Open email client</source>
        <translation>Abrir cliente de correo</translation>
    </message>
    <message>
        <source>Copied!</source>
        <translation>¡Copiado!</translation>
    </message>

    <!-- SSL / Antivirus error titles -->
    <message>
        <source>SSL Certificate Error</source>
        <translation>Error de certificado SSL</translation>
    </message>
    <message>
        <source>Installation Blocked</source>
        <translation>Instalación bloqueada</translation>
    </message>

    <message>
        <source>Click is outside the &apos;{layer}&apos; raster. To segment another raster, stop the current segmentation first.</source>
        <translation>El clic está fuera del raster '{layer}'. Para segmentar otro raster, detén primero la segmentación actual.</translation>
    </message>

    <!-- Update notification -->
    <message>
        <source>Big update dropped — v{version} is here!</source>
        <translation>Gran actualización — ¡la v{version} ya está aquí!</translation>
    </message>
    <message>
        <source>Grab it now</source>
        <translation>Actualiza ahora</translation>
    </message>

    <!-- Format conversion -->
    <message>
        <source>{ext} format is not directly supported. GDAL is not available.
Please convert your raster to GeoTIFF (.tif) before using AI Segmentation.</source>
        <translation>El formato {ext} no es compatible directamente. GDAL no está disponible.
Por favor, convierta su ráster a GeoTIFF (.tif) antes de usar AI Segmentation.</translation>
    </message>
    <message>
        <source>Cannot open {ext} file. The format may not be supported by your QGIS installation.
Please convert your raster to GeoTIFF (.tif) before using AI Segmentation.</source>
        <translation>No se puede abrir el archivo {ext}. El formato puede no ser compatible con su instalación de QGIS.
Por favor, convierta su ráster a GeoTIFF (.tif) antes de usar AI Segmentation.</translation>
    </message>
    <message>
        <source>Failed to read {ext} file: {error}
Please convert your raster to GeoTIFF (.tif) manually.</source>
        <translation>Error al leer el archivo {ext}: {error}
Por favor, convierta su ráster a GeoTIFF (.tif) manualmente.</translation>
    </message>

    <!-- PyTorch DLL Error Messages -->
    <message>
        <source>PyTorch Error</source>
        <translation>Error de PyTorch</translation>
    </message>
    <message>
        <source>PyTorch cannot load on Windows</source>
        <translation>PyTorch no puede cargarse en Windows</translation>
    </message>
    <message>
        <source>The plugin requires Visual C++ Redistributables to run PyTorch.

Please download and install:
https://aka.ms/vs/17/release/vc_redist.x64.exe

After installation, restart QGIS and try again.</source>
        <translation>El plugin requiere Visual C++ Redistributables para ejecutar PyTorch.

Por favor, descargue e instale:
https://aka.ms/vs/17/release/vc_redist.x64.exe

Después de la instalación, reinicie QGIS e intente nuevamente.</translation>
    </message>
    <message>
        <source>Prediction Error</source>
        <translation>Error de predicción</translation>
    </message>
    <message>
        <source>Segmentation failed</source>
        <translation>Falló la segmentación</translation>
    </message>
    <message>
        <source>Layer data provider is not available.</source>
        <translation>El proveedor de datos de la capa no esta disponible.</translation>
    </message>
    <message>
        <source>Failed to fetch tiles from the online layer. Check your network connection.</source>
        <translation>Error al obtener tiles de la capa en linea. Verifique su conexion de red.</translation>
    </message>
    <message>
        <source>Online layer returned blank tiles for this area. Try panning to an area with data coverage.</source>
        <translation>La capa en linea devolvio tiles en blanco para esta area. Intente mover el mapa a una area con cobertura de datos.</translation>
    </message>
    <message>
        <source>Crop Error</source>
        <translation>Error de recorte</translation>
    </message>
    <message>
        <source>No raster file path available. Please restart segmentation.</source>
        <translation>No hay ruta de archivo raster disponible. Reinicie la segmentacion.</translation>
    </message>
    <message>
        <source>Encoding Error</source>
        <translation>Error de codificacion</translation>
    </message>
    <message>
        <source>Report a Bug</source>
        <translation>Reportar un Bug</translation>
    </message>
    <message>
        <source>Disconnected parts detected in your polygon.</source>
        <translation>Partes desconectadas detectadas en tu poligono.</translation>
    </message>
    <message>
        <source>For best accuracy, segment one element at a time.</source>
        <translation>Para mejor precision, segmenta un elemento a la vez.</translation>
    </message>
    <message>
        <source>Layer: {}</source>
        <translation>Capa: {}</translation>
    </message>
    <message>
        <source>Polygon saved! Click on another element to segment, or export your polygons.</source>
        <translation>Polígono guardado! Haz clic en otro elemento para segmentar, o exporta tus polígonos.</translation>
    </message>
    <message>
        <source>Disconnected parts detected. For best accuracy, segment one element at a time.</source>
        <translation>Partes desconectadas detectadas. Para mayor precisión, segmenta un elemento a la vez.</translation>
    </message>

    <message>
        <source>No element detected at this point. Try clicking on a different area.</source>
        <translation>No se detectó ningún elemento en este punto. Intenta hacer clic en otra área.</translation>
    </message>

    <message>
        <source>Updating...</source>
        <translation>Actualizando...</translation>
    </message>

    <message>
        <source>Check for Updates</source>
        <translation>Buscar actualizaciones</translation>
    </message>

    <message>
        <source>More from TerraLab...</source>
        <translation>Mas sobre TerraLab...</translation>
    </message>

    <message>
        <source>Missing Visual C++ Redistributable. Install it, restart your computer, then click Retry.</source>
        <translation>Falta Visual C++ Redistributable. Instalelo, reinicie su computadora y haga clic en Reintentar.</translation>
    </message>

    <message>
        <source>Retry</source>
        <translation>Reintentar</translation>
    </message>
<!-- v1.0.0 strings -->
    <message>
        <source>Cannot Write Export</source>
        <translation>No se puede exportar</translation>
    </message>
    <message>
        <source>Cannot create export directory '{path}': {reason}</source>
        <translation>No se puede crear el directorio de exportacion '{path}': {reason}</translation>
    </message>
    <message>
        <source>The export directory '{path}' is not writable. Choose a different location.</source>
        <translation>El directorio de exportacion '{path}' no permite escritura. Elige otra ubicacion.</translation>
    </message>
    <message>
        <source>Loading AI model...</source>
        <translation>Cargando el modelo IA...</translation>
    </message>
    <message>
        <source>SAM model ready</source>
        <translation>Modelo SAM listo</translation>
    </message>
    <message>
        <source>Ready</source>
        <translation>Listo</translation>
    </message>
    <message>
        <source>Model load failed</source>
        <translation>Fallo al cargar el modelo</translation>
    </message>
    <message>
        <source>Click landed outside the current element — segment one element at a time. Saving the current selection and starting a new one.</source>
        <translation>El clic quedo fuera del elemento actual — segmenta un elemento a la vez. Se guarda la seleccion actual y empieza una nueva.</translation>
    </message>
    <message>
        <source>New here?</source>
        <translation>Nuevo aqui?</translation>
    </message>
    <message>
        <source>Watch the tutorial</source>
        <translation>Ver el tutorial</translation>
    </message>
    <message>
        <source>Network Connection Problem</source>
        <translation>Problema de conexion de red</translation>
    </message>
    <message>
        <source>Your connection appears unstable or blocked. Check: (1) your internet is working, (2) QGIS > Settings > Options > Network has a proxy configured if you are on a corporate network, (3) your firewall allows connections to pypi.org and files.pythonhosted.org.</source>
        <translation>Tu conexion parece inestable o bloqueada. Comprueba: (1) que Internet funciona, (2) que QGIS > Configuracion > Opciones > Red tiene un proxy si estas en una red corporativa, (3) que tu cortafuegos permite conexiones a pypi.org y files.pythonhosted.org.</translation>
    </message>
    <message>
        <source>Sign in to TerraLab</source>
        <translation>Iniciar sesion en TerraLab</translation>
    </message>
    <message>
        <source>Two steps to start using AI Segmentation</source>
        <translation>Dos pasos para empezar a usar AI Segmentation</translation>
    </message>
    <message>
        <source>1. Sign up or sign in on terra-lab.ai to get your key</source>
        <translation>1. Regístrate o inicia sesión en terra-lab.ai para obtener tu clave</translation>
    </message>
    <message>
        <source>2. Paste your key below to activate</source>
        <translation>2. Pega tu clave abajo para activar</translation>
    </message>
    <message>
        <source>1. Sign up / Sign in</source>
        <translation>1. Regístrate / Inicia sesión</translation>
    </message>
    <message>
        <source>Get Your Key</source>
        <translation>Obtener tu clave</translation>
    </message>
    <message>
        <source>2. Paste your activation key</source>
        <translation>2. Pega tu clave de activación</translation>
    </message>
    <message>
        <source>Sign in to get your key</source>
        <translation>Inicia sesion para obtener tu clave</translation>
    </message>
    <message>
        <source>Create your free TerraLab account or sign in, then copy your activation key from the dashboard.</source>
        <translation>Crea tu cuenta gratuita de TerraLab o inicia sesion, luego copia tu clave de activacion desde el panel.</translation>
    </message>
    <message>
        <source>Activate</source>
        <translation>Activar</translation>
    </message>
    <message>
        <source>Please enter your activation key.</source>
        <translation>Por favor, introduce tu clave de activacion.</translation>
    </message>
    <message>
        <source>Checking...</source>
        <translation>Verificando...</translation>
    </message>
    <message>
        <source>Activation key verified!</source>
        <translation>Clave de activacion verificada!</translation>
    </message>
    <message>
        <source>Invalid activation key.</source>
        <translation>Clave de activacion invalida.</translation>
    </message>
    <message>
        <source>Cannot reach server. Check your internet connection.</source>
        <translation>No se puede conectar al servidor. Comprueba tu conexion a Internet.</translation>
    </message>
    <message>
        <source>Signed in!</source>
        <translation>Sesion iniciada!</translation>
    </message>
    <message>
        <source>AI Edit</source>
        <translation>AI Edit</translation>
    </message>
    <message>
        <source>Generate imagery with AI on map zones (opens AI Edit plugin)</source>
        <translation>Genera imagenes IA en zonas del mapa (abre el plugin AI Edit)</translation>
    </message>
    <message>
        <source>Right-click must be inside the current selection area.</source>
        <translation>El clic derecho debe estar dentro del area de seleccion actual.</translation>
    </message>
    <!-- Account Settings Dialog -->
    <message>
        <source>Account Settings</source>
        <translation>Configuracion de la cuenta</translation>
    </message>
    <message>
        <source>Loading account info...</source>
        <translation>Cargando informacion de la cuenta...</translation>
    </message>
    <message>
        <source>Manage account on terra-lab.ai</source>
        <translation>Gestionar cuenta en terra-lab.ai</translation>
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
        <translation>Cambiar clave de activacion</translation>
    </message>
    <message>
        <source>Plan</source>
        <translation>Plan</translation>
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
        <translation>Clave</translation>
    </message>
</context>
</TS>
