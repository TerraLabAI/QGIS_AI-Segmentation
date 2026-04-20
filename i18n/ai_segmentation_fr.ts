<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="fr_FR">
<context>
    <name>AISegmentation</name>
    <!-- Welcome Section -->
    <message>
        <source>Click Install to set up AI Segmentation</source>
        <translation>Cliquez sur Installer pour configurer AI Segmentation</translation>
    </message>

    <!-- Setup Section -->
    <message>
        <source>Installing AI Segmentation...</source>
        <translation>Installation de AI Segmentation...</translation>
    </message>
    <message>
        <source>Verifying installation...</source>
        <translation>Verification de l'installation...</translation>
    </message>
    <message>
        <source>Detecting device...</source>
        <translation>Detection de l'appareil...</translation>
    </message>
    <message>
        <source>Install path: {}</source>
        <translation>Dossier d'installation : {}</translation>
    </message>
    <message>
        <source>To install in a different folder, set the environment variable AI_SEGMENTATION_CACHE_DIR:</source>
        <translation>Pour installer dans un autre dossier, definissez la variable d'environnement AI_SEGMENTATION_CACHE_DIR :</translation>
    </message>
    <message>
        <source>1. Open Windows Settings &gt; System &gt; Advanced system settings
2. Click 'Environment Variables'
3. Under 'User variables', click 'New'
4. Variable name: AI_SEGMENTATION_CACHE_DIR
5. Variable value: the folder path you want to use
6. Click OK and restart QGIS</source>
        <translation>1. Ouvrir Parametres Windows &gt; Systeme &gt; Parametres systeme avances
2. Cliquer sur 'Variables d'environnement'
3. Sous 'Variables utilisateur', cliquer sur 'Nouveau'
4. Nom de la variable : AI_SEGMENTATION_CACHE_DIR
5. Valeur de la variable : le chemin du dossier souhaite
6. Cliquer OK et redemarrer QGIS</translation>
    </message>
    <message>
        <source>Run this command in Terminal, then restart QGIS:

launchctl setenv AI_SEGMENTATION_CACHE_DIR /your/path</source>
        <translation>Executez cette commande dans le Terminal, puis redemarrez QGIS :

launchctl setenv AI_SEGMENTATION_CACHE_DIR /votre/chemin</translation>
    </message>
    <message>
        <source>Add this line to your ~/.bashrc or ~/.profile, then restart QGIS:

export AI_SEGMENTATION_CACHE_DIR=/your/path</source>
        <translation>Ajoutez cette ligne dans votre ~/.bashrc ou ~/.profile, puis redemarrez QGIS :

export AI_SEGMENTATION_CACHE_DIR=/votre/chemin</translation>
    </message>
    <message>
        <source>Checking...</source>
        <translation>Verification...</translation>
    </message>
    <message>
        <source>Install</source>
        <translation>Installer</translation>
    </message>
    <message>
        <source>Update</source>
        <translation>Mettre a jour</translation>
    </message>
    <message>
        <source>Dependencies ready</source>
        <translation>Dependances pretes</translation>
    </message>
    <message>
        <source>Downloading AI model...</source>
        <translation>Telechargement du modele IA...</translation>
    </message>
    <message>
        <source>Dependencies ready, model not downloaded</source>
        <translation>Dependances pretes, modele non telecharge</translation>
    </message>
    <message>
        <source>Dependencies ready, model download failed</source>
        <translation>Dependances pretes, echec du telechargement du modele</translation>
    </message>
    <message>
        <source>Download Model</source>
        <translation>Telecharger le modele</translation>
    </message>
    <message>
        <source>Cancel</source>
        <translation>Annuler</translation>
    </message>
    <message>
        <source>Cancel installation</source>
        <translation>Annuler l'installation</translation>
    </message>
    <message>
        <source>Are you sure you want to cancel the installation?</source>
        <translation>Voulez-vous vraiment annuler l'installation ?</translation>
    </message>
    <message>
        <source>Installation cancelled</source>
        <translation>Installation annulée</translation>
    </message>
    <message>
        <source>Installation failed</source>
        <translation>Échec de l'installation</translation>
    </message>
    <message>
        <source>Verification failed:</source>
        <translation>Échec de la vérification :</translation>
    </message>
    <message>
        <source>Verification Failed</source>
        <translation>Échec de la vérification</translation>
    </message>
    <message>
        <source>Virtual environment was created but verification failed:</source>
        <translation>L'environnement virtuel a été créé mais la vérification a échoué :</translation>
    </message>
    <message>
        <source>Unknown error</source>
        <translation>Erreur inconnue</translation>
    </message>
    <message>
        <source>Installation Failed</source>
        <translation>Échec de l'installation</translation>
    </message>

    <!-- Model Section -->
    <message>
        <source>Update QGIS to 3.34+ for the latest AI model</source>
        <translation>Mettez QGIS a jour vers 3.34+ pour le dernier modele IA</translation>
    </message>
    <message>
        <source>Intel Mac: using SAM1 (compatible with PyTorch 2.2)</source>
        <translation>Mac Intel : utilisation de SAM1 (compatible avec PyTorch 2.2)</translation>
    </message>
    <message>
        <source>Download Failed</source>
        <translation>Échec du téléchargement</translation>
    </message>
    <message>
        <source>Failed to download model:</source>
        <translation>Impossible de télécharger le modèle :</translation>
    </message>

    <!-- Panel Title -->
    <message>
        <source>AI Segmentation by TerraLab</source>
        <translation>AI Segmentation par TerraLab</translation>
    </message>

    <!-- Segmentation Section -->
    <message>
        <source>Select a Raster Layer to Segment:</source>
        <translation>Sélectionnez un raster à segmenter :</translation>
    </message>
    <message>
        <source>Select a raster layer (GeoTIFF, WMS, XYZ tiles, etc.)</source>
        <translation>Sélectionnez une couche raster (GeoTIFF, WMS, tuiles XYZ, etc.)</translation>
    </message>
    <message>
        <source>No raster layer found. Add a GeoTIFF, image file, or online layer (WMS, XYZ) to your project.</source>
        <translation>Aucune couche raster trouvée. Ajoutez un GeoTIFF, un fichier image ou une couche en ligne (WMS, XYZ) a votre projet.</translation>
    </message>
    <message>
        <source>No layer found. Add a raster or online layer to your project.</source>
        <translation>Aucune couche trouvée. Ajoutez une couche raster ou en ligne à votre projet.</translation>
    </message>
    <message>
        <source>Start AI Segmentation</source>
        <translation>Démarrer AI Segmentation</translation>
    </message>
    <message>
        <source>Save polygon</source>
        <translation>Sauvegarder le polygone</translation>
    </message>
    <message>
        <source>Undo last point</source>
        <translation>Annuler le dernier point</translation>
    </message>
    <message>
        <source>Stop segmentation</source>
        <translation>Arrêter la segmentation</translation>
    </message>
    <message>
        <source>Segmentation</source>
        <translation>Segmentation</translation>
    </message>
    <message>
        <source>Navigation</source>
        <translation>Navigation</translation>
    </message>
    <message>
        <source>Space</source>
        <translation>Espace</translation>
    </message>
    <message>
        <source>Hold and move to pan the map</source>
        <translation>Maintenir et bouger pour déplacer la carte</translation>
    </message>
    <message>
        <source>Middle mouse button</source>
        <translation>Bouton central de la souris</translation>
    </message>
    <message>
        <source>Click and drag to pan the map</source>
        <translation>Cliquer et glisser pour déplacer la carte</translation>
    </message>
    <message>
        <source>Shortcuts</source>
        <translation>Raccourcis</translation>
    </message>
    <message>
        <source>Save current polygon to your session</source>
        <translation>Sauvegarder le polygone actuel dans votre session</translation>
    </message>
    <message>
        <source>The AI model works best on one element at a time.</source>
        <translation>Le modele IA fonctionne mieux sur un element a la fois.</translation>
    </message>
    <message>
        <source>Save your polygon before selecting the next element.</source>
        <translation>Sauvegardez votre polygone avant de selectionner le suivant.</translation>
    </message>
    <message>
        <source>Export polygon to a layer</source>
        <translation>Exporter le polygone vers une couche</translation>
    </message>
    <message>
        <source>Export {count} polygons to a layer</source>
        <translation>Exporter {count} polygones vers une couche</translation>
    </message>

    <!-- Refine Section -->
    <message>
        <source>Refine selection</source>
        <translation>Affiner la sélection</translation>
    </message>
    <message>
        <source>Expand/Contract:</source>
        <translation>Dilater/Éroder :</translation>
    </message>
    <message>
        <source>Positive = expand outward, Negative = shrink inward</source>
        <translation>Positif = dilater, Négatif = éroder</translation>
    </message>
    <message>
        <source>Simplify outline:</source>
        <translation>Simplifier le contour :</translation>
    </message>
    <message>
        <source>Reduce small variations in the outline (0 = no change)</source>
        <translation>Réduire les petites variations du contour (0 = aucun changement)</translation>
    </message>
    <message>
        <source>Fill holes:</source>
        <translation>Remplir les trous :</translation>
    </message>
    <message>
        <source>Fill interior holes in the selection</source>
        <translation>Combler les trous intérieurs de la sélection</translation>
    </message>
    <message>
        <source>Min area:</source>
        <translation>Surface min :</translation>
    </message>
    <message>
        <source>Remove polygons smaller than this area (in pixels)</source>
        <translation>Supprimer les polygones plus petits que cette surface (en pixels)</translation>
    </message>
    <message>
        <source>Round corners:</source>
        <translation>Arrondir les coins :</translation>
    </message>
    <message>
        <source>Round corners for natural shapes like trees and bushes. Increase 'Simplify outline' for smoother results.</source>
        <translation>Arrondir les coins pour des formes naturelles comme les arbres et buissons. Augmentez 'Simplifier le contour' pour un résultat plus lisse.</translation>
    </message>
    <message>
        <source>Outline</source>
        <translation>Contour</translation>
    </message>
    <message>
        <source>Selection</source>
        <translation>Sélection</translation>
    </message>

    <!-- Instructions -->
    <message>
        <source>Click on the element you want to segment:</source>
        <translation>Cliquez sur l'élément à segmenter :</translation>
    </message>
    <message>
        <source>Left-click to select</source>
        <translation>Clic gauche pour sélectionner</translation>
    </message>
    <message>
        <source>Left-click to add more</source>
        <translation>Clic gauche pour en ajouter</translation>
    </message>
    <message>
        <source>Right-click to exclude from selection</source>
        <translation>Clic droit pour exclure de la sélection</translation>
    </message>
    <message>
        <source>Invalid Layer</source>
        <translation>Couche invalide</translation>
    </message>
    <message>
        <source>Layer extent contains invalid coordinates (NaN/Inf). Check the raster file.</source>
        <translation>L'étendue de la couche contient des coordonnées invalides (NaN/Inf). Vérifiez le fichier raster.</translation>
    </message>

    <!-- Dialogs -->
    <message>
        <source>Not Ready</source>
        <translation>Pas prêt</translation>
    </message>
    <message>
        <source>Please wait for the SAM model to load.</source>
        <translation>Veuillez attendre le chargement du modèle SAM.</translation>
    </message>
    <message>
        <source>Load Failed</source>
        <translation>Échec du chargement</translation>
    </message>
    <message>
        <source>Layer Creation Failed</source>
        <translation>Échec de création de la couche</translation>
    </message>
    <message>
        <source>Could not create the output layer.</source>
        <translation>Impossible de créer la couche de sortie.</translation>
    </message>
    <message>
        <source>Export Failed</source>
        <translation>Échec de l'export</translation>
    </message>
    <message>
        <source>Could not save layer to file:</source>
        <translation>Impossible de sauvegarder la couche :</translation>
    </message>
    <message>
        <source>Layer was saved but could not be loaded:</source>
        <translation>La couche a été sauvegardée mais n'a pas pu être chargée :</translation>
    </message>
    <message>
        <source>You have {count} unsaved polygon(s).</source>
        <translation>Vous avez {count} polygone(s) non exporté(s).</translation>
    </message>
    <message>
        <source>Changing layer will discard your current segmentation. Continue?</source>
        <translation>Changer de couche supprimera votre segmentation en cours. Continuer ?</translation>
    </message>
    <message>
        <source>Change Layer?</source>
        <translation>Changer de couche ?</translation>
    </message>
    <message>
        <source>Stop Segmentation?</source>
        <translation>Arrêter la segmentation ?</translation>
    </message>
    <message>
        <source>This will discard {count} polygon(s).</source>
        <translation>Cela va supprimer {count} polygone(s).</translation>
    </message>
    <message>
        <source>Use 'Export to layer' to keep them.</source>
        <translation>Utilisez « Exporter en couche » pour les conserver.</translation>
    </message>
    <message>
        <source>This will end the current segmentation session.</source>
        <translation>Cela va mettre fin à la session de segmentation en cours.</translation>
    </message>
    <message>
        <source>Do you want to continue?</source>
        <translation>Voulez-vous continuer ?</translation>
    </message>
    <message>
        <source>Edit saved polygon</source>
        <translation>Modifier un polygone sauvegardé</translation>
    </message>
    <message>
        <source>Warning: you are about to edit an already saved polygon.</source>
        <translation>Attention : vous allez modifier un polygone déjà sauvegardé.</translation>
    </message>
    <message>
        <source>New to AI Segmentation?</source>
        <translation>Nouveau sur AI Segmentation ?</translation>
    </message>
    <message>
        <source>Watch our tutorial</source>
        <translation>Voir notre tutoriel</translation>
    </message>

    <!-- About Section -->
    <message>
        <source>Contact us</source>
        <translation>Nous contacter</translation>
    </message>
    <message>
        <source>Bug, question, feature request?</source>
        <translation>Bug, question, demande de fonctionnalité ?</translation>
    </message>
    <message>
        <source>We'd love to hear from you!</source>
        <translation>On serait ravis d'avoir de vos nouvelles !</translation>
    </message>
    <message>
        <source>Copy email address</source>
        <translation>Copier l'adresse email</translation>
    </message>
    <message>
        <source>or</source>
        <translation>ou</translation>
    </message>
    <message>
        <source>Book a video call</source>
        <translation>Réserver un appel vidéo</translation>
    </message>
    <message>
        <source>Tutorial</source>
        <translation>Tutoriel</translation>
    </message>
    <message>
        <source>Something not working?</source>
        <translation>Quelque chose ne fonctionne pas ?</translation>
    </message>
    <message>
        <source>Copy your logs and send them to us, we'll look into it :)</source>
        <translation>Copiez vos logs et envoyez-les nous, on regarde ça :)</translation>
    </message>

    <!-- Tooltip -->
    <message>
        <source>Segment elements on raster images using AI</source>
        <translation>Segmenter des elements sur des images raster avec l'IA</translation>
    </message>

    <!-- Error Report Dialog -->
    <message>
        <source>Copy your logs with the button below and send them to our email.</source>
        <translation>Copiez vos logs avec le bouton ci-dessous et envoyez-les à notre email.</translation>
    </message>
    <message>
        <source>We'll fix your issue :)</source>
        <translation>On va corriger votre problème :)</translation>
    </message>
    <message>
        <source>1. Click to copy logs</source>
        <translation>1. Cliquer pour copier les logs</translation>
    </message>
    <message>
        <source>2. Click to send to {}</source>
        <translation>2. Cliquer pour envoyer à {}</translation>
    </message>
    <message>
        <source>Open email client</source>
        <translation>Ouvrir le client email</translation>
    </message>
    <message>
        <source>Copied!</source>
        <translation>Copié !</translation>
    </message>

    <!-- SSL / Antivirus error titles -->
    <message>
        <source>SSL Certificate Error</source>
        <translation>Erreur de certificat SSL</translation>
    </message>
    <message>
        <source>Installation Blocked</source>
        <translation>Installation bloquée</translation>
    </message>

    <message>
        <source>Click is outside the &apos;{layer}&apos; raster. To segment another raster, stop the current segmentation first.</source>
        <translation>Le clic est en dehors du raster '{layer}'. Pour segmenter un autre raster, arrêtez d'abord la segmentation en cours.</translation>
    </message>

    <!-- Update notification -->
    <message>
        <source>Big update dropped — v{version} is here!</source>
        <translation>Grosse mise à jour — la v{version} est là !</translation>
    </message>
    <message>
        <source>Grab it now</source>
        <translation>Fonce mettre à jour</translation>
    </message>

    <!-- Format conversion -->
    <message>
        <source>{ext} format is not directly supported. GDAL is not available.
Please convert your raster to GeoTIFF (.tif) before using AI Segmentation.</source>
        <translation>Le format {ext} n'est pas directement supporté. GDAL n'est pas disponible.
Veuillez convertir votre raster en GeoTIFF (.tif) avant d'utiliser AI Segmentation.</translation>
    </message>
    <message>
        <source>Cannot open {ext} file. The format may not be supported by your QGIS installation.
Please convert your raster to GeoTIFF (.tif) before using AI Segmentation.</source>
        <translation>Impossible d'ouvrir le fichier {ext}. Le format n'est peut-être pas supporté par votre installation QGIS.
Veuillez convertir votre raster en GeoTIFF (.tif) avant d'utiliser AI Segmentation.</translation>
    </message>
    <message>
        <source>Failed to read {ext} file: {error}
Please convert your raster to GeoTIFF (.tif) manually.</source>
        <translation>Impossible de lire le fichier {ext} : {error}
Veuillez convertir votre raster en GeoTIFF (.tif) manuellement.</translation>
    </message>

    <!-- PyTorch DLL Error Messages -->
    <message>
        <source>PyTorch Error</source>
        <translation>Erreur PyTorch</translation>
    </message>
    <message>
        <source>PyTorch cannot load on Windows</source>
        <translation>PyTorch ne peut pas se charger sur Windows</translation>
    </message>
    <message>
        <source>The plugin requires Visual C++ Redistributables to run PyTorch.

Please download and install:
https://aka.ms/vs/17/release/vc_redist.x64.exe

After installation, restart QGIS and try again.</source>
        <translation>Le plugin nécessite Visual C++ Redistributables pour exécuter PyTorch.

Veuillez télécharger et installer :
https://aka.ms/vs/17/release/vc_redist.x64.exe

Après l'installation, redémarrez QGIS et réessayez.</translation>
    </message>
    <message>
        <source>Prediction Error</source>
        <translation>Erreur de prédiction</translation>
    </message>
    <message>
        <source>Segmentation failed</source>
        <translation>Échec de la segmentation</translation>
    </message>
    <message>
        <source>Layer data provider is not available.</source>
        <translation>Le fournisseur de données de la couche n'est pas disponible.</translation>
    </message>
    <message>
        <source>Failed to fetch tiles from the online layer. Check your network connection.</source>
        <translation>Impossible de recuperer les tuiles de la couche en ligne. Verifiez votre connexion reseau.</translation>
    </message>
    <message>
        <source>Online layer returned blank tiles for this area. Try panning to an area with data coverage.</source>
        <translation>La couche en ligne a retourne des tuiles vides pour cette zone. Essayez de deplacer la carte vers une zone avec des donnees.</translation>
    </message>
    <message>
        <source>Crop Error</source>
        <translation>Erreur d'extraction</translation>
    </message>
    <message>
        <source>No raster file path available. Please restart segmentation.</source>
        <translation>Aucun chemin de fichier raster disponible. Veuillez relancer la segmentation.</translation>
    </message>
    <message>
        <source>Encoding Error</source>
        <translation>Erreur d'encodage</translation>
    </message>
    <message>
        <source>Report a Bug</source>
        <translation>Signaler un bug</translation>
    </message>
    <message>
        <source>Disconnected parts detected in your polygon.</source>
        <translation>Parties deconnectees detectees dans votre polygone.</translation>
    </message>
    <message>
        <source>For best accuracy, segment one element at a time.</source>
        <translation>Pour une meilleure precision, segmentez un element a la fois.</translation>
    </message>
    <message>
        <source>Layer: {}</source>
        <translation>Couche : {}</translation>
    </message>
    <message>
        <source>Polygon saved! Click on another element to segment, or export your polygons.</source>
        <translation>Polygone enregistré ! Cliquez sur un autre élément à segmenter, ou exportez vos polygones.</translation>
    </message>
    <message>
        <source>Disconnected parts detected. For best accuracy, segment one element at a time.</source>
        <translation>Parties déconnectées détectées. Pour une meilleure précision, segmentez un élément à la fois.</translation>
    </message>

    <message>
        <source>No element detected at this point. Try clicking on a different area.</source>
        <translation>Aucun élément détecté à cet endroit. Essayez de cliquer sur une autre zone.</translation>
    </message>

    <message>
        <source>Updating...</source>
        <translation>Mise a jour...</translation>
    </message>

    <message>
        <source>Check for Updates</source>
        <translation>Rechercher des mises a jour</translation>
    </message>

    <message>
        <source>More from TerraLab...</source>
        <translation>En savoir plus sur TerraLab...</translation>
    </message>

    <message>
        <source>Missing Visual C++ Redistributable. Install it, restart your computer, then click Retry.</source>
        <translation>Visual C++ Redistributable manquant. Installez-le, redemarrez votre ordinateur, puis cliquez sur Reessayer.</translation>
    </message>

    <message>
        <source>Retry</source>
        <translation>Reessayer</translation>
    </message>
<!-- v1.0.0 strings -->
    <message>
        <source>Cannot Write Export</source>
        <translation>Impossible d exporter</translation>
    </message>
    <message>
        <source>Cannot create export directory '{path}': {reason}</source>
        <translation>Impossible de creer le dossier d export '{path}' : {reason}</translation>
    </message>
    <message>
        <source>The export directory '{path}' is not writable. Choose a different location.</source>
        <translation>Le dossier d export '{path}' n est pas accessible en ecriture. Choisissez un autre emplacement.</translation>
    </message>
    <message>
        <source>Loading AI model...</source>
        <translation>Chargement du modele IA...</translation>
    </message>
    <message>
        <source>SAM model ready</source>
        <translation>Modele SAM pret</translation>
    </message>
    <message>
        <source>Ready</source>
        <translation>Pret</translation>
    </message>
    <message>
        <source>Model load failed</source>
        <translation>Echec du chargement du modele</translation>
    </message>
    <message>
        <source>Click landed outside the current element — segment one element at a time. Saving the current selection and starting a new one.</source>
        <translation>Le clic se trouve hors de l element courant — segmentez un element a la fois. La selection en cours est enregistree, un nouvel element commence.</translation>
    </message>
    <message>
        <source>New here?</source>
        <translation>Nouveau ici ?</translation>
    </message>
    <message>
        <source>Watch the tutorial</source>
        <translation>Voir le tutoriel</translation>
    </message>
    <message>
        <source>Network Connection Problem</source>
        <translation>Probleme de connexion reseau</translation>
    </message>
    <message>
        <source>Your connection appears unstable or blocked. Check: (1) your internet is working, (2) QGIS > Settings > Options > Network has a proxy configured if you are on a corporate network, (3) your firewall allows connections to pypi.org and files.pythonhosted.org.</source>
        <translation>Votre connexion semble instable ou bloquee. Verifiez : (1) que votre Internet fonctionne, (2) que QGIS > Reglages > Options > Reseau contient un proxy si vous etes sur un reseau d entreprise, (3) que votre pare-feu autorise les connexions a pypi.org et files.pythonhosted.org.</translation>
    </message>
    <message>
        <source>Sign in to TerraLab</source>
        <translation>Se connecter a TerraLab</translation>
    </message>
    <message>
        <source>Sign in to continue</source>
        <translation>Connectez-vous pour continuer</translation>
    </message>
    <message>
        <source>Create your free TerraLab account or sign in to get started.</source>
        <translation>Creez votre compte TerraLab gratuit ou connectez-vous pour commencer.</translation>
    </message>
    <message>
        <source>Sign in to TerraLab (free)</source>
        <translation>Se connecter a TerraLab (gratuit)</translation>
    </message>
    <message>
        <source>Sign in to get your key</source>
        <translation>Connectez-vous pour obtenir votre cle</translation>
    </message>
    <message>
        <source>Create your free TerraLab account or sign in, then copy your activation key from the dashboard.</source>
        <translation>Creez votre compte TerraLab gratuit ou connectez-vous, puis copiez votre cle d activation depuis le tableau de bord.</translation>
    </message>
    <message>
        <source>Activate</source>
        <translation>Activer</translation>
    </message>
    <message>
        <source>Please enter your activation key.</source>
        <translation>Veuillez entrer votre cle d activation.</translation>
    </message>
    <message>
        <source>Checking...</source>
        <translation>Verification...</translation>
    </message>
    <message>
        <source>Activation key verified!</source>
        <translation>Cle d activation verifiee !</translation>
    </message>
    <message>
        <source>Invalid activation key.</source>
        <translation>Cle d activation invalide.</translation>
    </message>
    <message>
        <source>Cannot reach server. Check your internet connection.</source>
        <translation>Impossible de joindre le serveur. Verifiez votre connexion Internet.</translation>
    </message>
    <message>
        <source>Signed in!</source>
        <translation>Connecte !</translation>
    </message>
    <message>
        <source>AI Edit</source>
        <translation>AI Edit</translation>
    </message>
    <message>
        <source>Generate imagery with AI on map zones (opens AI Edit plugin)</source>
        <translation>Generez des images IA sur des zones de la carte (ouvre le plugin AI Edit)</translation>
    </message>
    <message>
        <source>Right-click must be inside the current selection area.</source>
        <translation>Le clic droit doit se situer dans la zone de selection courante.</translation>
    </message>
    <!-- Account Settings Dialog -->
    <message>
        <source>Account Settings</source>
        <translation>Parametres du compte</translation>
    </message>
    <message>
        <source>Loading account info...</source>
        <translation>Chargement des informations du compte...</translation>
    </message>
    <message>
        <source>Manage subscription on terra-lab.ai</source>
        <translation>Gerer l'abonnement sur terra-lab.ai</translation>
    </message>
    <message>
        <source>Show</source>
        <translation>Afficher</translation>
    </message>
    <message>
        <source>Hide</source>
        <translation>Masquer</translation>
    </message>
    <message>
        <source>Change activation key</source>
        <translation>Changer la cle d'activation</translation>
    </message>
    <message>
        <source>Plan</source>
        <translation>Plan</translation>
    </message>
    <message>
        <source>Free</source>
        <translation>Gratuit</translation>
    </message>
    <message>
        <source>Canceled</source>
        <translation>Annule</translation>
    </message>
    <message>
        <source>Email</source>
        <translation>Email</translation>
    </message>
    <message>
        <source>Key</source>
        <translation>Cle</translation>
    </message>
</context>
</TS>
