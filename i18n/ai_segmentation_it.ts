<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="it_IT">
<context>
    <name>AISegmentation</name>
    <!-- Review display colours: Normal / Confidence / Random (2026-07-01) -->
    <message>
        <source>Display colors:</source>
        <translation>Colori di visualizzazione:</translation>
    </message>
    <message>
        <source>Normal</source>
        <translation>Normale</translation>
    </message>
    <message>
        <source>Confidence</source>
        <translation>Confidenza</translation>
    </message>
    <message>
        <source>Random</source>
        <translation>Casuale</translation>
    </message>
    <message>
        <source>Outline</source>
        <translation>Contorno</translation>
    </message>
    <message>
        <source>How detections are coloured on the map (visual only): Normal outline, Confidence heatmap (green sure, red uncertain), or a random colour per object to tell them apart.</source>
        <translation>Indica come vengono colorati i rilevamenti sulla mappa (solo a livello visivo): contorno Normale, mappa di calore per Confidenza (verde sicuro, rosso incerto), oppure un colore casuale per ogni oggetto per distinguerli.</translation>
    </message>
    <!-- Retry: back to setup keeping inputs (2026-07-01) -->
    <message>
        <source>Retry</source>
        <translation>Riprova</translation>
    </message>
    <message>
        <source>Go back to your zone, references and settings to adjust and detect again. Nothing is saved.</source>
        <translation>Torna alla zona, ai riferimenti e alle impostazioni per modificarli e rilevare di nuovo. Non viene salvato nulla.</translation>
    </message>
    <!-- Reference example enlarge (2026-07-01) -->
    <message>
        <source>Click to enlarge</source>
        <translation>Clicca per ingrandire</translation>
    </message>
    <message>
        <source>This is exactly what the AI uses: your object with a little of its surroundings.</source>
        <translation>Questo è esattamente ciò che l'IA utilizza: il tuo oggetto con un po' del contesto circostante.</translation>
    </message>
    <!-- Automatic step-2 redesign: describe + example + detail -->
    <message>
        <source>Describe what to find</source>
        <translation>Descrivi cosa cercare</translation>
    </message>
    <message>
        <source>solar panel, building, tree…</source>
        <translation>pannello solare, edificio, albero…</translation>
    </message>
    <message>
        <source>1-2 words, English</source>
        <translation>1-2 parole, in inglese</translation>
    </message>
    <message>
        <source>optional</source>
        <translation>opzionale</translation>
    </message>
    <message>
        <source>Show an example</source>
        <translation>Mostra un esempio</translation>
    </message>
    <message>
        <source>Outline one object; the AI finds the rest. No good name for it? Examples alone work too.</source>
        <translation>Delimita un oggetto; l'IA trova il resto. Non hai un nome preciso? Bastano anche solo gli esempi.</translation>
    </message>
    <message>
        <source>Exclude a look-alike</source>
        <translation>Escludi un elemento simile</translation>
    </message>
    <message>
        <source>Now outline a look-alike to exclude, then click the first point to close.</source>
        <translation>Ora delimita un elemento simile da escludere, poi clicca sul primo punto per chiudere.</translation>
    </message>
    <message>
        <source>Your examples drive the search.</source>
        <translation>I tuoi esempi guidano la ricerca.</translation>
    </message>
    <message>
        <source>Too generic to name. Clear the box to search from your example alone, or type a concrete object.</source>
        <translation>Troppo generico da nominare. Svuota il campo per cercare solo dal tuo esempio, oppure digita un oggetto concreto.</translation>
    </message>
    <message>
        <source>Example match</source>
        <translation>Corrispondenza per esempio</translation>
    </message>
    <message>
        <source>Include</source>
        <translation>Includi</translation>
    </message>
    <message>
        <source>Exclude</source>
        <translation>Escludi</translation>
    </message>
    <message>
        <source>Mark an object to find more like it.</source>
        <translation>Segna un oggetto per trovarne altri simili.</translation>
    </message>
    <message>
        <source>Mark a false positive to drop things like it.</source>
        <translation>Segna un falso positivo per scartare elementi simili.</translation>
    </message>
    <message>
        <source>Draw on map</source>
        <translation>Disegna sulla mappa</translation>
    </message>
    <message>
        <source>Outline one object on the map; SAM finds all similar ones.</source>
        <translation>Delimita un oggetto sulla mappa; SAM trova tutti quelli simili.</translation>
    </message>
    <message>
        <source>Finer detail finds smaller objects.</source>
        <translation>Un dettaglio più fine trova oggetti più piccoli.</translation>
    </message>
    <message>
        <source>{n} object(s) detected</source>
        <translation>{n} oggetto(i) rilevato(i)</translation>
    </message>
    <message>
        <source>Adjust below, then export</source>
        <translation>Modifica le impostazioni qui sotto, poi effettua l'export</translation>
    </message>
    <message>
        <source>Refine in Manual mode</source>
        <translation>Affina in modalità Manuale</translation>
    </message>
    <message>
        <source>Some objects off? Refine them in Manual mode first.</source>
        <translation>Qualche oggetto non è preciso? Affinalo prima in modalità Manuale.</translation>
    </message>
    <message>
        <source>Now outline one object on the map, then double-click to finish.</source>
        <translation>Ora delimita un oggetto sulla mappa, poi fai doppio clic per terminare.</translation>
    </message>
    <message>
        <source>Now outline one false positive on the map, then double-click to finish.</source>
        <translation>Ora delimita un falso positivo sulla mappa, poi fai doppio clic per terminare.</translation>
    </message>
    <!-- Refine in Manual handoff -->
    <message>
        <source>Refine in Manual</source>
        <translation>Affina in Manuale</translation>
    </message>
    <message>
        <source>Open these detections in Manual mode to fix specific objects with point-and-click, then return here to Finish.</source>
        <translation>Apri questi rilevamenti in modalità Manuale per correggere oggetti specifici con un clic, poi torna qui per Terminare.</translation>
    </message>
    <message>
        <source>Refining Automatic results</source>
        <translation>Affinamento dei risultati Automatici</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Fine-tune the detections, then go back to review to export.</source>
        <translation>Ritocca i rilevamenti, poi torna alla revisione per l'export.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Editing this detection.</source>
        <translation>Modifica di questo rilevamento.</translation>
    </message>
    <message>
        <source>Editing this detection</source>
        <translation>Modifica di questo rilevamento</translation>
    </message>
    <message>
        <source>adds area</source>
        <translation>aggiunge superficie</translation>
    </message>
    <message>
        <source>removes area</source>
        <translation>rimuove superficie</translation>
    </message>
    <message>
        <source>keeps it (turns green)</source>
        <translation>lo conserva (diventa verde)</translation>
    </message>
    <message>
        <source>removes the object</source>
        <translation>rimuove l'oggetto</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Press S to keep it (turns green) · Delete removes it</source>
        <translation>Premi S per conservarlo (diventa verde) · Canc lo rimuove</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Click a blue detection to open it for editing.</source>
        <translation>Clicca su un rilevamento blu per modificarlo.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Left-click adds area, right-click removes it. Press S to keep it (turns green).</source>
        <translation>Il clic sinistro aggiunge superficie, il clic destro la rimuove. Premi S per conservarlo (diventa verde).</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>{kept} of {total} kept - 'Back to review' to export.</source>
        <translation>{kept} su {total} conservati - «Torna alla revisione» per l'export.</translation>
    </message>
    <message>
        <source>Back to review</source>
        <translation>Torna alla revisione</translation>
    </message>
    <message>
        <source>Finish or go back to review to switch modes.</source>
        <translation>Termina oppure torna alla revisione per cambiare modalità.</translation>
    </message>
    <message>
        <source>Finish or exit the review to switch modes.</source>
        <translation>Termina oppure esci dalla revisione per cambiare modalità.</translation>
    </message>
    <message>
        <source>Preparing Manual mode, loading the local model...</source>
        <translation>Preparazione della modalità Manuale, caricamento del modello locale...</translation>
    </message>
    <message>
        <source>Blue = detections to review, one at a time.</source>
        <translation>Blu = rilevamenti da rivedere, uno alla volta.</translation>
    </message>
    <message>
        <source>Left-click a detection to edit it (adds area); right-click to remove a part</source>
        <translation>Clic sinistro su un rilevamento per modificarlo (aggiunge superficie); clic destro per rimuoverne una parte</translation>
    </message>
    <message>
        <source>Press S to validate it (turns green), then move on to the next one.</source>
        <translation>Premi S per validarlo (diventa verde), poi passa al successivo.</translation>
    </message>
    <message>
        <source>Locked - refined in Manual mode</source>
        <translation>Bloccato - affinato in modalità Manuale</translation>
    </message>
    <message>
        <source>Confidence is locked while you refine in Manual mode.</source>
        <translation>La confidenza è bloccata durante l'affinamento in modalità Manuale.</translation>
    </message>
    <message>
        <source>Refining in Manual needs the local model. Open Manual mode once to finish setup, then try again.</source>
        <translation>L'affinamento in Manuale richiede il modello locale. Apri una volta la modalità Manuale per completare l'installazione, poi riprova.</translation>
    </message>
    <message>
        <source>Detection</source>
        <translation>Rilevamento</translation>
    </message>
    <message>
        <source>Confidence:</source>
        <translation>Confidenza:</translation>
    </message>
    <message>
        <source>Minimum confidence to keep a detected object. Lower finds more objects but may add false positives; raise it for cleaner results on large, distinct features.</source>
        <translation>Confidenza minima per conservare un oggetto rilevato. Un valore più basso trova più oggetti ma può aggiungere falsi positivi; aumentala per risultati più puliti su elementi grandi e distinti.</translation>
    </message>
    <!-- Account Settings: Dependencies -->
    <message>
        <source>Dependencies</source>
        <translation>Dipendenze</translation>
    </message>
    <message>
        <source>Local AI model files stored on this computer.</source>
        <translation>File del modello IA locale memorizzati su questo computer.</translation>
    </message>
    <message>
        <source>On disk</source>
        <translation>Su disco</translation>
    </message>
    <message>
        <source>Not installed</source>
        <translation>Non installato</translation>
    </message>
    <message>
        <source>Open folder</source>
        <translation>Apri cartella</translation>
    </message>
    <!-- v1.2 strings previously missing from translations -->
    <message>
        <source>Accept the Terms and Privacy Policy to enable segmentation.</source>
        <translation>Accetta i Termini e l'Informativa sulla privacy per attivare la segmentazione.</translation>
    </message>
    <message>
        <source>An unexpected error occurred during export. Please check the logs.</source>
        <translation>Si è verificato un errore imprevisto durante l'export. Controlla i log.</translation>
    </message>
    <message>
        <source>I agree to the &lt;a href=&quot;{terms}&quot;&gt;Terms&lt;/a&gt; and &lt;a href=&quot;{privacy}&quot;&gt;Privacy Policy&lt;/a&gt;</source>
        <translation>Accetto i &lt;a href=&quot;{terms}&quot;&gt;Termini&lt;/a&gt; e l'&lt;a href=&quot;{privacy}&quot;&gt;Informativa sulla privacy&lt;/a&gt;</translation>
    </message>
    <message>
        <source>No valid polygons could be created from the selection. Try adjusting the refine settings or making a new selection.</source>
        <translation>Non è stato possibile creare poligoni validi dalla selezione. Prova a modificare le impostazioni di affinamento o a fare una nuova selezione.</translation>
    </message>
    <message>
        <source>Repairing Installation</source>
        <translation>Riparazione dell'installazione</translation>
    </message>
    <message>
        <source>Repairing installation...</source>
        <translation>Riparazione dell'installazione...</translation>
    </message>
    <message>
        <source>Restart QGIS Required</source>
        <translation>Riavvio di QGIS richiesto</translation>
    </message>
    <message>
        <source>Something went wrong with this click, so it was not applied. Please try again.</source>
        <translation>Qualcosa è andato storto con questo clic, quindi non è stato applicato. Riprova.</translation>
    </message>
    <message>
        <source>The Python runtime used by the AI engine is damaged (this can be caused by a disk cleanup tool or antivirus). It will now be repaired automatically. Please try your selection again once the repair finishes.</source>
        <translation>L'ambiente Python utilizzato dal motore IA è danneggiato (può essere causato da uno strumento di pulizia disco o da un antivirus). Verrà ora riparato automaticamente. Riprova la tua selezione una volta terminata la riparazione.</translation>
    </message>
    <message>
        <source>Your polygons were added as a temporary layer so nothing is lost.</source>
        <translation>I tuoi poligoni sono stati aggiunti come livello temporaneo, così non si perde nulla.</translation>
    </message>
    <message>
        <source>Could not write to {name}. Saved to a separate file instead.</source>
        <translation>Impossibile scrivere in {name}. Salvato invece in un file separato.</translation>
    </message>
    <!-- Welcome Section -->
    <message>
        <source>Click Install to set up AI Segmentation</source>
        <translation>Clicca su Installa per configurare AI Segmentation</translation>
    </message>
    <!-- Setup Section -->
    <message>
        <source>Installing AI Segmentation...</source>
        <translation>Installazione di AI Segmentation...</translation>
    </message>
    <message>
        <source>Verifying installation...</source>
        <translation>Verifica dell'installazione...</translation>
    </message>
    <message>
        <source>Detecting device...</source>
        <translation>Rilevamento del dispositivo...</translation>
    </message>
    <message>
        <source>Install path: {}</source>
        <translation>Percorso di installazione: {}</translation>
    </message>
    <message>
        <source>To install in a different folder, set the environment variable AI_SEGMENTATION_CACHE_DIR:</source>
        <translation>Per installare in una cartella diversa, impostare la variabile d'ambiente AI_SEGMENTATION_CACHE_DIR:</translation>
    </message>
    <message>
        <source>1. Open Windows Settings &gt; System &gt; Advanced system settings
2. Click 'Environment Variables'
3. Under 'User variables', click 'New'
4. Variable name: AI_SEGMENTATION_CACHE_DIR
5. Variable value: the folder path you want to use
6. Click OK and restart QGIS</source>
        <translation>1. Aprire Impostazioni Windows &gt; Sistema &gt; Impostazioni avanzate di sistema
2. Fare clic su 'Variabili d'ambiente'
3. In 'Variabili utente', fare clic su 'Nuova'
4. Nome variabile: AI_SEGMENTATION_CACHE_DIR
5. Valore variabile: il percorso della cartella da utilizzare
6. Fare clic su OK e riavviare QGIS</translation>
    </message>
    <message>
        <source>Run this command in Terminal, then restart QGIS:

launchctl setenv AI_SEGMENTATION_CACHE_DIR /your/path</source>
        <translation>Eseguire questo comando nel Terminale, poi riavviare QGIS:

launchctl setenv AI_SEGMENTATION_CACHE_DIR /tuo/percorso</translation>
    </message>
    <message>
        <source>Add this line to your ~/.bashrc or ~/.profile, then restart QGIS:

export AI_SEGMENTATION_CACHE_DIR=/your/path</source>
        <translation>Aggiungere questa riga al proprio ~/.bashrc o ~/.profile, poi riavviare QGIS:

export AI_SEGMENTATION_CACHE_DIR=/tuo/percorso</translation>
    </message>
    <message>
        <source>Checking...</source>
        <translation>Verifica in corso...</translation>
    </message>
    <message>
        <source>Install</source>
        <translation>Installa</translation>
    </message>
    <message>
        <source>Update</source>
        <translation>Aggiorna</translation>
    </message>
    <message>
        <source>Dependencies ready</source>
        <translation>Dipendenze pronte</translation>
    </message>
    <message>
        <source>Downloading AI model...</source>
        <translation>Download del modello IA in corso...</translation>
    </message>
    <message>
        <source>Dependencies ready, model not downloaded</source>
        <translation>Dipendenze pronte, modello non scaricato</translation>
    </message>
    <message>
        <source>Dependencies ready, model download failed</source>
        <translation>Dipendenze pronte, download del modello non riuscito</translation>
    </message>
    <message>
        <source>Download Model</source>
        <translation>Scarica modello</translation>
    </message>
    <message>
        <source>Cancel</source>
        <translation>Annulla</translation>
    </message>
    <message>
        <source>Cancel installation</source>
        <translation>Annulla installazione</translation>
    </message>
    <message>
        <source>Are you sure you want to cancel the installation?</source>
        <translation>Sei sicuro di voler annullare l'installazione?</translation>
    </message>
    <message>
        <source>Installation cancelled</source>
        <translation>Installazione annullata</translation>
    </message>
    <message>
        <source>Installation failed</source>
        <translation>Installazione non riuscita</translation>
    </message>
    <message>
        <source>Verification failed:</source>
        <translation>Verifica non riuscita:</translation>
    </message>
    <message>
        <source>Verification Failed</source>
        <translation>Verifica non riuscita</translation>
    </message>
    <message>
        <source>Virtual environment was created but verification failed:</source>
        <translation>L'ambiente virtuale è stato creato ma la verifica non è riuscita:</translation>
    </message>
    <message>
        <source>Unknown error</source>
        <translation>Errore sconosciuto</translation>
    </message>
    <message>
        <source>Installation Failed</source>
        <translation>Installazione non riuscita</translation>
    </message>
    <!-- Model Section -->
    <message>
        <source>Update QGIS to 3.34+ for the latest AI model</source>
        <translation>Aggiorna QGIS alla versione 3.34+ per l'ultimo modello IA</translation>
    </message>
    <message>
        <source>Intel Mac: using SAM1 (compatible with PyTorch 2.2)</source>
        <translation>Mac Intel: utilizzo di SAM1 (compatibile con PyTorch 2.2)</translation>
    </message>
    <message>
        <source>Download Failed</source>
        <translation>Download non riuscito</translation>
    </message>
    <message>
        <source>Failed to download model:</source>
        <translation>Download del modello non riuscito:</translation>
    </message>
    <!-- Panel Title -->
    <message>
        <source>AI Segmentation by TerraLab</source>
        <translation>AI Segmentation by TerraLab</translation>
    </message>
    <!-- Segmentation Section -->
    <message>
        <source>Select a Raster Layer to Segment:</source>
        <translation>Seleziona un livello raster da segmentare:</translation>
    </message>
    <message>
        <source>Select a raster layer (GeoTIFF, WMS, XYZ tiles, etc.)</source>
        <translation>Seleziona un livello raster (GeoTIFF, WMS, tile XYZ, ecc.)</translation>
    </message>
    <message>
        <source>No raster layer found. Add a GeoTIFF, image file, or online layer (WMS, XYZ) to your project.</source>
        <translation>Nessun livello raster trovato. Aggiungi un GeoTIFF, un file immagine o un livello online (WMS, XYZ) al progetto.</translation>
    </message>
    <message>
        <source>No layer found. Add a raster or online layer to your project.</source>
        <translation>Nessun livello trovato. Aggiungi un livello raster o online al progetto.</translation>
    </message>
    <message>
        <source>Start AI Segmentation</source>
        <translation>Avvia AI Segmentation</translation>
    </message>
    <message>
        <source>Save polygon</source>
        <translation>Salva poligono</translation>
    </message>
    <message>
        <source>Undo last point</source>
        <translation>Annulla ultimo punto</translation>
    </message>
    <message>
        <source>Stop segmentation</source>
        <translation>Interrompi segmentazione</translation>
    </message>
    <message>
        <source>Segmentation</source>
        <translation>Segmentazione</translation>
    </message>
    <message>
        <source>Navigation</source>
        <translation>Navigazione</translation>
    </message>
    <message>
        <source>Space</source>
        <translation>Spazio</translation>
    </message>
    <message>
        <source>Hold and move to pan the map</source>
        <translation>Tieni premuto e sposta per scorrere la mappa</translation>
    </message>
    <message>
        <source>Middle mouse button</source>
        <translation>Tasto centrale del mouse</translation>
    </message>
    <message>
        <source>Click and drag to pan the map</source>
        <translation>Clicca e trascina per scorrere la mappa</translation>
    </message>
    <message>
        <source>Shortcuts</source>
        <translation>Scorciatoie</translation>
    </message>
    <message>
        <source>Save current polygon to your session</source>
        <translation>Salva il poligono attuale nella tua sessione</translation>
    </message>
    <message>
        <source>The AI model works best on one element at a time.</source>
        <translation>Il modello IA funziona meglio su un elemento alla volta.</translation>
    </message>
    <message>
        <source>Save your polygon before selecting the next element.</source>
        <translation>Salva il tuo poligono prima di selezionare l'elemento successivo.</translation>
    </message>
    <message>
        <source>Export polygon to a layer</source>
        <translation>Export poligono su un livello</translation>
    </message>
    <message>
        <source>Export {count} polygons to a layer</source>
        <translation>Export di {count} poligoni su un livello</translation>
    </message>
    <!-- Refine Section -->
    <message>
        <source>Refine selection</source>
        <translation>Affina selezione</translation>
    </message>
    <message>
        <source>Expand/Contract:</source>
        <translation>Dilata/Contrai:</translation>
    </message>
    <message>
        <source>Positive = expand outward, Negative = shrink inward</source>
        <translation>Positivo = dilata verso l'esterno, Negativo = contrai verso l'interno</translation>
    </message>
    <message>
        <source>Simplify outline:</source>
        <translation>Semplifica contorno:</translation>
    </message>
    <message>
        <source>Reduce small variations in the outline (0 = no change)</source>
        <translation>Riduce le piccole variazioni del contorno (0 = nessuna modifica)</translation>
    </message>
    <message>
        <source>Fill holes:</source>
        <translation>Riempi buchi:</translation>
    </message>
    <message>
        <source>Fill interior holes in the selection</source>
        <translation>Riempie i buchi interni della selezione</translation>
    </message>
    <message>
        <source>Min area:</source>
        <translation>Area min:</translation>
    </message>
    <message>
        <source>Remove polygons smaller than this area (in pixels)</source>
        <translation>Rimuove i poligoni più piccoli di questa superficie (in pixel)</translation>
    </message>
    <message>
        <source>Shape</source>
        <translation>Forma</translation>
    </message>
    <message>
        <source>Size</source>
        <translation>Dimensione</translation>
    </message>
    <message>
        <source>Round corners:</source>
        <translation>Arrotonda angoli:</translation>
    </message>
    <message>
        <source>Round corners for natural shapes like trees and bushes. Increase 'Simplify outline' for smoother results.</source>
        <translation>Arrotonda gli angoli per forme naturali come alberi e cespugli. Aumenta 'Semplifica contorno' per risultati più fluidi.</translation>
    </message>
    <message>
        <source>Outline</source>
        <translation>Contorno</translation>
    </message>
    <message>
        <source>Selection</source>
        <translation>Selezione</translation>
    </message>
    <!-- Instructions -->
    <message>
        <source>Click on the element you want to segment:</source>
        <translation>Clicca sull'elemento che vuoi segmentare:</translation>
    </message>
    <message>
        <source>Left-click to select</source>
        <translation>Clic sinistro per selezionare</translation>
    </message>
    <message>
        <source>Left-click to add more</source>
        <translation>Clic sinistro per aggiungerne altri</translation>
    </message>
    <message>
        <source>Right-click to exclude from selection</source>
        <translation>Clic destro per escludere dalla selezione</translation>
    </message>
    <message>
        <source>Invalid Layer</source>
        <translation>Livello non valido</translation>
    </message>
    <message>
        <source>Layer extent contains invalid coordinates (NaN/Inf). Check the raster file.</source>
        <translation>L'estensione del livello contiene coordinate non valide (NaN/Inf). Controlla il file raster.</translation>
    </message>
    <!-- Dialogs -->
    <message>
        <source>Not Ready</source>
        <translation>Non pronto</translation>
    </message>
    <message>
        <source>Please wait for the SAM model to load.</source>
        <translation>Attendi il caricamento del modello SAM.</translation>
    </message>
    <message>
        <source>Load Failed</source>
        <translation>Caricamento non riuscito</translation>
    </message>
    <message>
        <source>Layer Creation Failed</source>
        <translation>Creazione del livello non riuscita</translation>
    </message>
    <message>
        <source>Could not create the output layer.</source>
        <translation>Impossibile creare il livello di output.</translation>
    </message>
    <message>
        <source>Export Failed</source>
        <translation>Export non riuscito</translation>
    </message>
    <message>
        <source>Could not save layer to file:</source>
        <translation>Impossibile salvare il livello nel file:</translation>
    </message>
    <message>
        <source>Layer was saved but could not be loaded:</source>
        <translation>Il livello è stato salvato ma non è stato possibile caricarlo:</translation>
    </message>
    <message>
        <source>You have {count} unsaved polygon(s).</source>
        <translation>Hai {count} poligono(i) non salvato(i).</translation>
    </message>
    <message>
        <source>Changing layer will discard your current segmentation. Continue?</source>
        <translation>Cambiare livello eliminerà la segmentazione in corso. Continuare?</translation>
    </message>
    <message>
        <source>Change Layer?</source>
        <translation>Cambiare livello?</translation>
    </message>
    <message>
        <source>Stop Segmentation?</source>
        <translation>Interrompere la segmentazione?</translation>
    </message>
    <message>
        <source>This will discard {count} polygon(s).</source>
        <translation>Questo eliminerà {count} poligono(i).</translation>
    </message>
    <message>
        <source>Use 'Export to layer' to keep them.</source>
        <translation>Usa «Export su livello» per conservarli.</translation>
    </message>
    <message>
        <source>This will end the current segmentation session.</source>
        <translation>Questo terminerà la sessione di segmentazione in corso.</translation>
    </message>
    <message>
        <source>Do you want to continue?</source>
        <translation>Vuoi continuare?</translation>
    </message>
    <message>
        <source>Edit saved polygon</source>
        <translation>Modifica un poligono salvato</translation>
    </message>
    <message>
        <source>Warning: you are about to edit an already saved polygon.</source>
        <translation>Attenzione: stai per modificare un poligono già salvato.</translation>
    </message>
    <message>
        <source>New to AI Segmentation?</source>
        <translation>Nuovo su AI Segmentation?</translation>
    </message>
    <message>
        <source>Watch our tutorial</source>
        <translation>Guarda il nostro tutorial</translation>
    </message>
    <!-- About Section -->
    <message>
        <source>Contact us</source>
        <translation>Contattaci</translation>
    </message>
    <message>
        <source>Bug, question, feature request?</source>
        <translation>Bug, domanda, richiesta di funzionalità?</translation>
    </message>
    <message>
        <source>We'd love to hear from you!</source>
        <translation>Ci farebbe piacere sentirti!</translation>
    </message>
    <message>
        <source>Copy email address</source>
        <translation>Copia l'indirizzo email</translation>
    </message>
    <message>
        <source>or</source>
        <translation>oppure</translation>
    </message>
    <message>
        <source>Book a video call</source>
        <translation>Prenota una videochiamata</translation>
    </message>
    <message>
        <source>Tutorial</source>
        <translation>Tutorial</translation>
    </message>
    <message>
        <source>Settings</source>
        <translation>Impostazioni</translation>
    </message>
    <message>
        <source>Help</source>
        <translation>Aiuto</translation>
    </message>
    <message>
        <source>Terms</source>
        <translation>Termini</translation>
    </message>
    <message>
        <source>Privacy</source>
        <translation>Privacy</translation>
    </message>
    <message>
        <source>Something not working?</source>
        <translation>Qualcosa non funziona?</translation>
    </message>
    <message>
        <source>Copy your logs and send them to us, we'll look into it :)</source>
        <translation>Copia i tuoi log e inviaceli, ci penseremo noi :)</translation>
    </message>
    <!-- Tooltip -->
    <message>
        <source>Segment elements on raster images using AI</source>
        <translation>Segmenta elementi su immagini raster con l'IA</translation>
    </message>
    <!-- Error Report Dialog -->
    <message>
        <source>Copy your logs with the button below and send them to our email.</source>
        <translation>Copia i tuoi log con il pulsante qui sotto e inviali alla nostra email.</translation>
    </message>
    <message>
        <source>We'll fix your issue :)</source>
        <translation>Risolveremo il tuo problema :)</translation>
    </message>
    <message>
        <source>1. Click to copy logs</source>
        <translation>1. Clicca per copiare i log</translation>
    </message>
    <message>
        <source>2. Click to send to {}</source>
        <translation>2. Clicca per inviare a {}</translation>
    </message>
    <message>
        <source>Open email client</source>
        <translation>Apri client email</translation>
    </message>
    <message>
        <source>Copied!</source>
        <translation>Copiato!</translation>
    </message>
    <!-- SSL / Antivirus error titles -->
    <message>
        <source>SSL Certificate Error</source>
        <translation>Errore certificato SSL</translation>
    </message>
    <message>
        <source>Installation Blocked</source>
        <translation>Installazione bloccata</translation>
    </message>
    <message>
        <source>Click is outside the &apos;{layer}&apos; raster. To segment another raster, stop the current segmentation first.</source>
        <translation>Il clic è fuori dal raster '{layer}'. Per segmentare un altro raster, interrompi prima la segmentazione in corso.</translation>
    </message>
    <!-- Update notification -->
    <message>
        <source>Big update dropped — v{version} is here!</source>
        <translation>Grande aggiornamento in arrivo — la v{version} è qui!</translation>
    </message>
    <message>
        <source>Grab it now</source>
        <translation>Scaricala ora</translation>
    </message>
    <!-- Format conversion -->
    <message>
        <source>{ext} format is not directly supported. GDAL is not available.
Please convert your raster to GeoTIFF (.tif) before using AI Segmentation.</source>
        <translation>Il formato {ext} non è supportato direttamente. GDAL non è disponibile.
Converti il tuo raster in GeoTIFF (.tif) prima di usare AI Segmentation.</translation>
    </message>
    <message>
        <source>Cannot open {ext} file. The format may not be supported by your QGIS installation.
Please convert your raster to GeoTIFF (.tif) before using AI Segmentation.</source>
        <translation>Impossibile aprire il file {ext}. Il formato potrebbe non essere supportato dalla tua installazione di QGIS.
Converti il tuo raster in GeoTIFF (.tif) prima di usare AI Segmentation.</translation>
    </message>
    <message>
        <source>Failed to read {ext} file: {error}
Please convert your raster to GeoTIFF (.tif) manually.</source>
        <translation>Impossibile leggere il file {ext}: {error}
Converti manualmente il tuo raster in GeoTIFF (.tif).</translation>
    </message>
    <!-- PyTorch DLL Error Messages -->
    <message>
        <source>PyTorch Error</source>
        <translation>Errore PyTorch</translation>
    </message>
    <message>
        <source>PyTorch cannot load on Windows</source>
        <translation>PyTorch non può essere caricato su Windows</translation>
    </message>
    <message>
        <source>The plugin requires Visual C++ Redistributables to run PyTorch.

Please download and install:
https://aka.ms/vs/17/release/vc_redist.x64.exe

After installation, restart QGIS and try again.</source>
        <translation>Il plugin richiede Visual C++ Redistributables per eseguire PyTorch.

Scarica e installa:
https://aka.ms/vs/17/release/vc_redist.x64.exe

Dopo l'installazione, riavvia QGIS e riprova.</translation>
    </message>
    <message>
        <source>Prediction Error</source>
        <translation>Errore di predizione</translation>
    </message>
    <message>
        <source>Segmentation failed</source>
        <translation>Segmentazione non riuscita</translation>
    </message>
    <message>
        <source>Layer data provider is not available.</source>
        <translation>Il provider dei dati del livello non è disponibile.</translation>
    </message>
    <message>
        <source>Failed to fetch tiles from the online layer. Check your network connection.</source>
        <translation>Impossibile recuperare i tile dal livello online. Controlla la tua connessione di rete.</translation>
    </message>
    <message>
        <source>Online layer returned blank tiles for this area. Try panning to an area with data coverage.</source>
        <translation>Il livello online ha restituito tile vuoti per questa zona. Prova a spostarti verso un'area con copertura dati.</translation>
    </message>
    <message>
        <source>Crop Error</source>
        <translation>Errore di ritaglio</translation>
    </message>
    <message>
        <source>No raster file path available. Please restart segmentation.</source>
        <translation>Nessun percorso del file raster disponibile. Riavvia la segmentazione.</translation>
    </message>
    <message>
        <source>Encoding Error</source>
        <translation>Errore di codifica</translation>
    </message>
    <message>
        <source>Report a Bug</source>
        <translation>Segnala un bug</translation>
    </message>
    <message>
        <source>Disconnected parts detected in your polygon.</source>
        <translation>Sono state rilevate parti disconnesse nel tuo poligono.</translation>
    </message>
    <message>
        <source>For best accuracy, segment one element at a time.</source>
        <translation>Per una precisione migliore, segmenta un elemento alla volta.</translation>
    </message>
    <message>
        <source>Layer: {}</source>
        <translation>Livello: {}</translation>
    </message>
    <message>
        <source>Polygon saved! Click on another element to segment, or export your polygons.</source>
        <translation>Poligono salvato! Clicca su un altro elemento da segmentare, oppure fai l'export dei tuoi poligoni.</translation>
    </message>
    <message>
        <source>Disconnected parts detected. For best accuracy, segment one element at a time.</source>
        <translation>Rilevate parti disconnesse. Per una precisione migliore, segmenta un elemento alla volta.</translation>
    </message>
    <message>
        <source>No element detected at this point. Try clicking on a different area.</source>
        <translation>Nessun elemento rilevato in questo punto. Prova a cliccare su un'altra zona.</translation>
    </message>
    <message>
        <source>Updating...</source>
        <translation>Aggiornamento in corso...</translation>
    </message>
    <message>
        <source>Check for Updates</source>
        <translation>Controlla aggiornamenti</translation>
    </message>
    <message>
        <source>More from TerraLab...</source>
        <translation>Altro da TerraLab...</translation>
    </message>
    <message>
        <source>Missing Visual C++ Redistributable. Install it, restart your computer, then click Retry.</source>
        <translation>Visual C++ Redistributable mancante. Installalo, riavvia il computer, poi clicca su Riprova.</translation>
    </message>
    <message>
        <source>Retry</source>
        <translation>Riprova</translation>
    </message>
    <!-- v1.0.0 strings -->
    <message>
        <source>Cannot Write Export</source>
        <translation>Impossibile scrivere l'Export</translation>
    </message>
    <message>
        <source>Cannot create export directory '{path}': {reason}</source>
        <translation>Impossibile creare la cartella di export '{path}': {reason}</translation>
    </message>
    <message>
        <source>The export directory '{path}' is not writable. Choose a different location.</source>
        <translation>La cartella di export '{path}' non è scrivibile. Scegli un'altra posizione.</translation>
    </message>
    <message>
        <source>Loading AI model...</source>
        <translation>Caricamento del modello IA...</translation>
    </message>
    <message>
        <source>SAM model ready</source>
        <translation>Modello SAM pronto</translation>
    </message>
    <message>
        <source>Ready</source>
        <translation>Pronto</translation>
    </message>
    <message>
        <source>Model load failed</source>
        <translation>Caricamento del modello non riuscito</translation>
    </message>
    <message>
        <source>Click landed outside the current element — segment one element at a time. Saving the current selection and starting a new one.</source>
        <translation>Il clic è caduto fuori dall'elemento corrente — segmenta un elemento alla volta. La selezione attuale viene salvata e ne inizia una nuova.</translation>
    </message>
    <message>
        <source>New here?</source>
        <translation>Sei nuovo qui?</translation>
    </message>
    <message>
        <source>Watch the tutorial</source>
        <translation>Guarda il tutorial</translation>
    </message>
    <message>
        <source>Network Connection Problem</source>
        <translation>Problema di connessione di rete</translation>
    </message>
    <message>
        <source>Your connection appears unstable or blocked. Check: (1) your internet is working, (2) QGIS > Settings > Options > Network has a proxy configured if you are on a corporate network, (3) your firewall allows connections to pypi.org and files.pythonhosted.org.</source>
        <translation>La tua connessione sembra instabile o bloccata. Verifica: (1) che la tua connessione Internet funzioni, (2) che QGIS > Impostazioni > Opzioni > Rete abbia un proxy configurato se sei su una rete aziendale, (3) che il tuo firewall consenta le connessioni a pypi.org e files.pythonhosted.org.</translation>
    </message>
    <message>
        <source>Sign in to TerraLab</source>
        <translation>Accedi a TerraLab</translation>
    </message>
    <message>
        <source>Two steps to start using AI Segmentation</source>
        <translation>Due passaggi per iniziare a usare AI Segmentation</translation>
    </message>
    <message>
        <source>1. Sign up or sign in on terra-lab.ai to get your key</source>
        <translation>1. Registrati o accedi su terra-lab.ai per ottenere la tua chiave</translation>
    </message>
    <message>
        <source>2. Paste your key below to activate</source>
        <translation>2. Incolla la tua chiave qui sotto per attivare</translation>
    </message>
    <message>
        <source>1. Sign up / Sign in</source>
        <translation>1. Registrazione / Accesso</translation>
    </message>
    <message>
        <source>Get Your Key</source>
        <translation>Ottieni la tua chiave</translation>
    </message>
    <message>
        <source>2. Paste your activation key</source>
        <translation>2. Incolla la tua chiave di attivazione</translation>
    </message>
    <message>
        <source>Sign in to get your key</source>
        <translation>Accedi per ottenere la tua chiave</translation>
    </message>
    <message>
        <source>Create your free TerraLab account or sign in, then copy your activation key from the dashboard.</source>
        <translation>Crea il tuo account TerraLab gratuito o accedi, poi copia la tua chiave di attivazione dalla dashboard.</translation>
    </message>
    <message>
        <source>Activate</source>
        <translation>Attiva</translation>
    </message>
    <message>
        <source>Please enter your activation key.</source>
        <translation>Inserisci la tua chiave di attivazione.</translation>
    </message>
    <message>
        <source>Checking...</source>
        <translation>Verifica in corso...</translation>
    </message>
    <message>
        <source>Activation key verified!</source>
        <translation>Chiave di attivazione verificata!</translation>
    </message>
    <message>
        <source>Invalid activation key.</source>
        <translation>Chiave di attivazione non valida.</translation>
    </message>
    <message>
        <source>Cannot reach server. Check your internet connection.</source>
        <translation>Impossibile contattare il server. Controlla la tua connessione Internet.</translation>
    </message>
    <message>
        <source>Signed in!</source>
        <translation>Accesso effettuato!</translation>
    </message>
    <message>
        <source>AI Edit</source>
        <translation>AI Edit</translation>
    </message>
    <message>
        <source>Generate imagery with AI on map zones (opens AI Edit plugin)</source>
        <translation>Genera immagini con l'IA su zone della mappa (apre il plugin AI Edit)</translation>
    </message>
    <message>
        <source>Right-click must be inside the current selection area.</source>
        <translation>Il clic destro deve essere all'interno dell'area di selezione corrente.</translation>
    </message>
    <!-- Account Settings Dialog -->
    <message>
        <source>Account Settings</source>
        <translation>Impostazioni account</translation>
    </message>
    <message>
        <source>Loading account info...</source>
        <translation>Caricamento delle informazioni dell'account...</translation>
    </message>
    <message>
        <source>Manage account on terra-lab.ai</source>
        <translation>Gestisci l'account su terra-lab.ai</translation>
    </message>
    <message>
        <source>Show</source>
        <translation>Mostra</translation>
    </message>
    <message>
        <source>Hide</source>
        <translation>Nascondi</translation>
    </message>
    <message>
        <source>Change activation key</source>
        <translation>Cambia chiave di attivazione</translation>
    </message>
    <message>
        <source>Plan</source>
        <translation>Piano</translation>
    </message>
    <message>
        <source>Free</source>
        <translation>Gratuito</translation>
    </message>
    <message>
        <source>Canceled</source>
        <translation>Annullato</translation>
    </message>
    <message>
        <source>Email</source>
        <translation>Email</translation>
    </message>
    <message>
        <source>Key</source>
        <translation>Chiave</translation>
    </message>
    <!-- Corrupt checkpoint recovery (#65) -->
    <message>
        <source>Model File Corrupted</source>
        <translation>File del modello corrotto</translation>
    </message>
    <message>
        <source>Re-downloading Model</source>
        <translation>Nuovo download del modello</translation>
    </message>
    <message>
        <source>The AI model file was corrupted and is being re-downloaded. Please try your selection again once it finishes.</source>
        <translation>Il file del modello IA era corrotto ed è in fase di nuovo download. Riprova la tua selezione una volta terminato.</translation>
    </message>
    <message>
        <source>The AI model file is corrupted but could not be removed automatically. Please delete this folder and restart QGIS:</source>
        <translation>Il file del modello IA è corrotto ma non è stato possibile rimuoverlo automaticamente. Elimina questa cartella e riavvia QGIS:</translation>
    </message>
    <!-- One-click sign-in (browser pairing, mirrors AI Edit) -->
    <message>
        <source>Segment your map with AI</source>
        <translation>Segmenta la tua mappa con l'IA</translation>
    </message>
    <message>
        <source>Sign in / Sign up to start</source>
        <translation>Accedi / Registrati per iniziare</translation>
    </message>
    <message>
        <source>Sign in via your browser to start using AI Segmentation</source>
        <translation>Accedi tramite il tuo browser per iniziare a usare AI Segmentation</translation>
    </message>
    <message>
        <source>Open again</source>
        <translation>Riapri</translation>
    </message>
    <message>
        <source>Didn't open? Open the page again</source>
        <translation>Non si è aperto? Riapri la pagina</translation>
    </message>
    <message>
        <source>Couldn't open your browser. Use the manual key option below.</source>
        <translation>Impossibile aprire il tuo browser. Usa l'opzione della chiave manuale qui sotto.</translation>
    </message>
    <message>
        <source>Sign-in timed out. Click Connect to try again.</source>
        <translation>Accesso scaduto. Clicca su Connetti per riprovare.</translation>
    </message>
    <message>
        <source>Sign-in was cancelled in the browser. Click Connect to try again.</source>
        <translation>L'accesso è stato annullato nel browser. Clicca su Connetti per riprovare.</translation>
    </message>
    <message>
        <source>Unexpected response from the server. Please try again.</source>
        <translation>Risposta inattesa dal server. Riprova.</translation>
    </message>
    <message>
        <source>This account has no active AI Segmentation plan. Reactivate it on terra-lab.ai, then click Connect again.</source>
        <translation>Questo account non ha un piano AI Segmentation attivo. Riattivalo su terra-lab.ai, poi clicca di nuovo su Connetti.</translation>
    </message>
    <message>
        <source>Connecting AI Segmentation</source>
        <translation>Connessione di AI Segmentation</translation>
    </message>
    <message>
        <source>Cancelling sign-in</source>
        <translation>Annullamento dell'accesso</translation>
    </message>
    <!-- Help menu / account settings (mirrors AI Edit) -->
    <message>
        <source>Help / Report a problem</source>
        <translation>Aiuto / Segnala un problema</translation>
    </message>
    <message>
        <source>Report a problem</source>
        <translation>Segnala un problema</translation>
    </message>
    <message>
        <source>Connected</source>
        <translation>Connesso</translation>
    </message>
    <message>
        <source>Sign out</source>
        <translation>Disconnetti</translation>
    </message>
    <message>
        <source>Sign out of AI Segmentation?</source>
        <translation>Disconnettersi da AI Segmentation?</translation>
    </message>
    <message>
        <source>You can sign back in anytime from QGIS.</source>
        <translation>Puoi accedere di nuovo in qualsiasi momento da QGIS.</translation>
    </message>
    <message>
        <source>Active</source>
        <translation>Attivo</translation>
    </message>
    <message>
        <source>Free Trial</source>
        <translation>Prova gratuita</translation>
    </message>
    <message>
        <source>Make this map presentation-ready</source>
        <translation>Rendi questa mappa pronta per la presentazione</translation>
    </message>
    <message>
        <source>AI Edit: turn your imagery into presentation and planning visuals</source>
        <translation>AI Edit: trasforma le tue immagini in visual di presentazione e pianificazione</translation>
    </message>
    <!-- Pro / Automatic mode strings (plan #79) -->
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Mode selection</source>
        <translation>Selezione modalità</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Choose between Manual (local) and Automatic (cloud) segmentation</source>
        <translation>Scegli tra la segmentazione Manuale (locale) e Automatica (cloud)</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Stop the active segmentation before switching modes.</source>
        <translation>Interrompi la segmentazione attiva prima di cambiare modalità.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Cancel the active detection before switching modes.</source>
        <translation>Annulla il rilevamento attivo prima di cambiare modalità.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Sign in to use Automatic mode</source>
        <translation>Accedi per usare la modalità Automatica</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Your free detections are used up</source>
        <translation>I tuoi rilevamenti gratuiti sono esauriti</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Subscribe to keep detecting without limits:</source>
        <translation>Abbonati per continuare a rilevare senza limiti:</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Subscribe to Pro</source>
        <translation>Abbonati a Pro</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detect every building, tree, or road automatically</source>
        <translation>Rileva automaticamente ogni edificio, albero o strada</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>No installation required, no GPU, no limits</source>
        <translation>Nessuna installazione richiesta, nessuna GPU, nessun limite</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Built for large-scale digitization projects</source>
        <translation>Pensato per progetti di digitalizzazione su larga scala</translation>
    </message>
    <message>
        <location filename="../src/ui/zone_selection_maptool.py" line="0"/>
        <source>Clear this zone</source>
        <translation>Elimina questa zona</translation>
    </message>
    <message>
        <location filename="../src/ui/zone_selection_maptool.py" line="0"/>
        <source>Cancel the running detection first</source>
        <translation>Annulla prima il rilevamento in corso</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>What do you want to detect?</source>
        <translation>Cosa vuoi rilevare?</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Where should the AI look?</source>
        <translation>Dove deve guardare l'IA?</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Change</source>
        <translation>Modifica</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>What to detect...</source>
        <translation>Cosa rilevare...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Hold the left mouse button and drag to draw a box on the map.</source>
        <translation>Tieni premuto il tasto sinistro del mouse e trascina per disegnare un riquadro sulla mappa.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} tile(s) = {n} credit(s)</source>
        <translation>{n} tile = {n} credito(i)</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Object: {obj}</source>
        <translation>Oggetto: {obj}</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detecting "{obj}"...</source>
        <translation>Rilevamento di "{obj}"...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Ground resolution per pixel. A smaller value lets the model detect smaller objects.</source>
        <translation>Risoluzione al suolo per pixel. Un valore più piccolo permette al modello di rilevare oggetti più piccoli.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detail</source>
        <translation>Dettaglio</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Higher detail splits the zone into more tiles. Each tile costs 1 credit and captures smaller objects.</source>
        <translation>Un dettaglio più alto divide la zona in più tile. Ogni tile costa 1 credito e cattura oggetti più piccoli.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Zone too large - reduce the selection area</source>
        <translation>Zona troppo grande - riduci l'area di selezione</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detect objects</source>
        <translation>Rileva oggetti</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Cancel detection</source>
        <translation>Annulla rilevamento</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Tile {current}/{total}</source>
        <translation>Tile {current}/{total}</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Sending to the AI…</source>
        <translation>Invio all'IA…</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>You're next · starting now…</source>
        <translation>Sei il prossimo · avvio in corso…</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Spot reserved · starting in ~{eta}</source>
        <translation>Posto riservato · avvio in ~{eta}</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Spot reserved · starting soon…</source>
        <translation>Posto riservato · avvio a breve…</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>High demand · your spot is held…</source>
        <translation>Alta richiesta · il tuo posto è riservato…</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Spot reserved · starting in a few seconds…</source>
        <translation>Posto riservato · avvio in pochi secondi…</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{s} seconds</source>
        <translation>{s} secondi</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{m} min</source>
        <translation>{m} min</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} credits remaining (resets {date})</source>
        <translation>{n} crediti rimanenti (si rinnova il {date})</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} credits remaining</source>
        <translation>{n} crediti rimanenti</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} free detection(s) remaining</source>
        <translation>{n} rilevamento(i) gratuito(i) rimanente(i)</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Drawing...</source>
        <translation>Disegno in corso...</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>{n} free detection(s) remaining (lifetime)</source>
        <translation>{n} rilevamento(i) gratuito(i) rimanente(i) (a vita)</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Upgrade to Pro on terra-lab.ai</source>
        <translation>Passa a Pro su terra-lab.ai</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Pro</source>
        <translation>Pro</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>{remaining} / {total} credits</source>
        <translation>{remaining} / {total} crediti</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>resets {date}</source>
        <translation>si rinnova il {date}</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Free uses</source>
        <translation>Utilizzi gratuiti</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Credits</source>
        <translation>Crediti</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Auto detection (live)</source>
        <translation>Rilevamento automatico (in corso)</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Preparing tiles...</source>
        <translation>Preparazione dei tile...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Could not render the zone. Try a smaller area or another layer.</source>
        <translation>Impossibile visualizzare la zona. Prova un'area più piccola o un altro livello.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Cancelling...</source>
        <translation>Annullamento...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Finishing the previous run, please wait a moment...</source>
        <translation>Conclusione dell'esecuzione precedente, attendi un momento...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Found {n} object(s) but could not save the result file. Check folder permissions and the log.</source>
        <translation>Trovato(i) {n} oggetto(i) ma impossibile salvare il file dei risultati. Controlla i permessi della cartella e il log.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Could not save the result file. Check the log.</source>
        <translation>Impossibile salvare il file dei risultati. Controlla il log.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>No detections found. Try a different prompt or zoom level.</source>
        <translation>Nessun rilevamento trovato. Prova un prompt diverso o un altro livello di zoom.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Resume detection</source>
        <translation>Riprendi rilevamento</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Detection failed. Check your connection and try again.</source>
        <translation>Rilevamento non riuscito. Controlla la connessione e riprova.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Not enough credits to continue. The finished tiles are kept.</source>
        <translation>Crediti insufficienti per continuare. I tile completati vengono conservati.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Zone too large. Reduce the area to 50 tiles or fewer.</source>
        <translation>Zona troppo grande. Riduci l'area a 50 tile o meno.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Automatic detection is temporarily unavailable. Please try again later.</source>
        <translation>Il rilevamento automatico è temporaneamente non disponibile. Riprova più tardi.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Draw a zone first. Automatic detection on online layers needs a zone.</source>
        <translation>Disegna prima una zona. Il rilevamento automatico sui livelli online richiede una zona.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>The zone is outside the selected raster layer. Pick the right layer or redraw the zone.</source>
        <translation>La zona è fuori dal livello raster selezionato. Scegli il livello corretto o ridisegna la zona.</translation>
    </message>
    <message>
        <source>Next</source>
        <translation>Avanti</translation>
    </message>
    <message>
        <source>Export to layer</source>
        <translation>Export su livello</translation>
    </message>
    <message>
        <source>{n} object(s) detected - adjust below, then export</source>
        <translation>{n} oggetto(i) rilevato(i) - modifica qui sotto, poi effettua l'export</translation>
    </message>
    <message>
        <source>Exported {n} polygon(s) to {name}</source>
        <translation>Export di {n} poligono(i) su {name} completato</translation>
    </message>
    <message>
        <source>Round corners</source>
        <translation>Arrotonda angoli</translation>
    </message>
    <message>
        <source>Fill holes</source>
        <translation>Riempi buchi</translation>
    </message>
    <message>
        <source>Expand/Shrink:</source>
        <translation>Dilata/Contrai:</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Less</source>
        <translation>Meno</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>More</source>
        <translation>Più</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>This area is large for this detail level. Raise detail or zoom in for sharper detections.</source>
        <translation>Questa area è grande per questo livello di dettaglio. Aumenta il dettaglio o esegui uno zoom per rilevamenti più precisi.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>This zone is too large for sharp detections, even at maximum detail. Draw a smaller zone for the best results.</source>
        <translation>Questa zona è troppo grande per rilevamenti precisi, anche al massimo dettaglio. Disegna una zona più piccola per i migliori risultati.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Available when detection finishes</source>
        <translation>Disponibile al termine del rilevamento</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Finishing up... {done}/{total}</source>
        <translation>Finalizzazione... {done}/{total}</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Min size:</source>
        <translation>Dimensione min:</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Max size:</source>
        <translation>Dimensione max:</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Off</source>
        <translation>Disattivato</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>No limit</source>
        <translation>Nessun limite</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Hide detections smaller than this ground area. Use it to drop tiny noise blobs. 0 = keep all.</source>
        <translation>Nasconde i rilevamenti più piccoli di questa superficie al suolo. Utile per scartare piccoli artefatti di rumore. 0 = mantieni tutto.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Hide detections larger than this ground area. 0 = no limit.</source>
        <translation>Nasconde i rilevamenti più grandi di questa superficie al suolo. 0 = nessun limite.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Finer detail finds smaller objects and costs more credits.</source>
        <translation>Un dettaglio più fine trova oggetti più piccoli e costa più crediti.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>≈ {n} credits</source>
        <translation>≈ {n} crediti</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Finish</source>
        <translation>Termina</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} object(s) detected - adjust below, then Finish</source>
        <translation>{n} oggetto(i) rilevato(i) - modifica qui sotto, poi Termina</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Out of credits. Keep what was found below, then Finish.</source>
        <translation>Crediti esauriti. Conserva ciò che è stato trovato qui sotto, poi Termina.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Saved {n} polygon(s) to {name}</source>
        <translation>{n} poligono(i) salvato(i) in {name}</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Cannot reach the server. Check your internet connection.</source>
        <translation>Impossibile contattare il server. Controlla la tua connessione Internet.</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Server refused the connection.</source>
        <translation>Il server ha rifiutato la connessione.</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Request timed out. Check your connection or try again.</source>
        <translation>Richiesta scaduta. Controlla la connessione o riprova.</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>SSL certificate error. Your network may be blocking secure connections.</source>
        <translation>Errore del certificato SSL. La tua rete potrebbe bloccare le connessioni sicure.</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Proxy connection failed. Check QGIS proxy settings (Settings &gt; Options &gt; Network).</source>
        <translation>Connessione al proxy non riuscita. Controlla le impostazioni proxy di QGIS (Impostazioni &gt; Opzioni &gt; Rete).</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Authentication failed. Please sign in again.</source>
        <translation>Autenticazione non riuscita. Accedi di nuovo.</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Network error. Check your internet connection.</source>
        <translation>Errore di rete. Controlla la tua connessione Internet.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Checking your AI Segmentation subscription</source>
        <translation>Verifica del tuo abbonamento AI Segmentation</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Loading AI Segmentation settings</source>
        <translation>Caricamento delle impostazioni di AI Segmentation</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Refreshing credits</source>
        <translation>Aggiornamento dei crediti</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Warming up AI Segmentation</source>
        <translation>Avvio di AI Segmentation</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>What do you want to segment?</source>
        <translation>Cosa vuoi segmentare?</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>e.g. building, solar panel (in English)</source>
        <translation>es. building, solar panel (in inglese)</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Popular</source>
        <translation>Popolari</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Library</source>
        <translation>Libreria</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Browse objects with before / after examples.</source>
        <translation>Esplora gli oggetti con esempi prima / dopo.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>The prompt is sent to the AI in English. Describe the object in 1-2 words (e.g. building, solar panel).</source>
        <translation>Il prompt viene inviato all'IA in inglese. Descrivi l'oggetto in 1-2 parole (es. building, solar panel).</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use just 1-2 words for the object.</source>
        <translation>Usa solo 1-2 parole per l'oggetto.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Type the object itself, not a sentence or question.</source>
        <translation>Digita direttamente l'oggetto, non una frase o una domanda.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Too generic. Draw an example instead, or use a concrete word like building.</source>
        <translation>Troppo generico. Disegna invece un esempio, oppure usa una parola concreta come building.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Name a concrete object, not how it looks.</source>
        <translation>Nomina un oggetto concreto, non il suo aspetto.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Segment one object - drop words like 'near' or 'with'.</source>
        <translation>Segmenta un solo oggetto - elimina parole come «near» o «with».</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use a real object word.</source>
        <translation>Usa una vera parola di oggetto.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use a 1-2 word object name.</source>
        <translation>Usa un nome di oggetto di 1-2 parole.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Did you mean '{term}'?</source>
        <translation>Intendevi '{term}'?</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Loading...</source>
        <translation>Caricamento...</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>No preview</source>
        <translation>Nessuna anteprima</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>No preview yet</source>
        <translation>Ancora nessuna anteprima</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Segment library</source>
        <translation>Libreria di segmentazione</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>RECENT</source>
        <translation>RECENTI</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Recently detected</source>
        <translation>Rilevati di recente</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>BROWSE</source>
        <translation>ESPLORA</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Objects you detect will appear here.</source>
        <translation>Gli oggetti che rilevi appariranno qui.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>today</source>
        <translation>oggi</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>yesterday</source>
        <translation>ieri</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>{n} days ago</source>
        <translation>{n} giorni fa</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>{n} detection(s)</source>
        <translation>{n} rilevamento(i)</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>{n} object(s)</source>
        <translation>{n} oggetto(i)</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Loading segment library</source>
        <translation>Caricamento della libreria di segmentazione</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Pick an object to see a before / after, then use it.</source>
        <translation>Scegli un oggetto per vedere un prima / dopo, poi usalo.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Search objects... e.g. building, solar panel</source>
        <translation>Cerca oggetti... es. building, solar panel</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Prompt:</source>
        <translation>Prompt:</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Fuzzy edges: this one may need cleanup after detection.</source>
        <translation>Bordi imprecisi: questo potrebbe richiedere una pulizia dopo il rilevamento.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use this prompt</source>
        <translation>Usa questo prompt</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>No matching objects.</source>
        <translation>Nessun oggetto corrispondente.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use</source>
        <translation>Usa</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>≈ {n} tiles = {n} credits</source>
        <translation>≈ {n} tile = {n} crediti</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Automatic mode scans your zone tile by tile. 1 tile = 1 credit, so this run costs about {n} credits. More detail splits the zone into more tiles, which costs more credits.</source>
        <translation>La modalità Automatica analizza la tua zona tile per tile. 1 tile = 1 credito, quindi questa esecuzione costa circa {n} crediti. Più dettaglio divide la zona in più tile, il che costa più crediti.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Draw your example inside the selected zone.</source>
        <translation>Disegna il tuo esempio all'interno della zona selezionata.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Clean edges:</source>
        <translation>Pulisci bordi:</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Remove thin ragged fringe attached to the outline (0 = no change)</source>
        <translation>Rimuove la sottile frangia irregolare attaccata al contorno (0 = nessuna modifica)</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Share anonymous usage statistics</source>
        <translation>Condividi statistiche di utilizzo anonime</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Helps us fix bugs faster. Never includes your data, layers or coordinates.</source>
        <translation>Ci aiuta a correggere i bug più velocemente. Non include mai i tuoi dati, livelli o coordinate.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} objects found</source>
        <translation>{n} oggetti trovati</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>No objects found</source>
        <translation>Nessun oggetto trovato</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>All shown at {pct}% confidence</source>
        <translation>Tutti mostrati al {pct}% di confidenza</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{visible} shown at {pct}% · {hidden} more below this confidence</source>
        <translation>{visible} mostrati al {pct}% · altri {hidden} sotto questa confidenza</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>0 shown at {pct}% - lower Confidence to reveal them</source>
        <translation>0 mostrati al {pct}% - abbassa la Confidenza per rivelarli</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Started at {pct}% - nothing scored above.</source>
        <translation>Avviato al {pct}% - nulla ha un punteggio superiore.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>More objects</source>
        <translation>Più oggetti</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Only confident</source>
        <translation>Solo i sicuri</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Objects not quite right? Refine them in Manual mode.</source>
        <translation>Oggetti non del tutto corretti? Affinali in modalità Manuale.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Export {n} polygons</source>
        <translation>Export di {n} poligoni</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Lower Confidence to show objects first.</source>
        <translation>Abbassa la Confidenza per mostrare prima gli oggetti.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Adjust &amp; run again</source>
        <translation>Modifica e riavvia</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Discard these detections?</source>
        <translation>Scartare questi rilevamenti?</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Your {total} detections will be discarded. You keep your zone, object and settings. Running Detect again will use new credits.</source>
        <translation>I tuoi {total} rilevamenti verranno scartati. Mantieni la zona, l'oggetto e le impostazioni. Rilanciare Rileva utilizzerà nuovi crediti.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Discard &amp;&amp; adjust</source>
        <translation>Scarta &amp;&amp; modifica</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Keep your detections?</source>
        <translation>Conservare i tuoi rilevamenti?</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Save {visible} detections to a layer before leaving?</source>
        <translation>Salvare {visible} rilevamenti in un livello prima di uscire?</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Save {total} detections (currently hidden by Confidence) to a layer before leaving?</source>
        <translation>Salvare {total} rilevamenti (attualmente nascosti dalla Confidenza) in un livello prima di uscire?</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Save &amp;&amp; exit</source>
        <translation>Salva &amp;&amp; esci</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Discard &amp;&amp; exit</source>
        <translation>Scarta &amp;&amp; esci</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>No {object} found in this zone</source>
        <translation>Nessun {object} trovato in questa zona</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>This run used {n} credits. Things that usually fix it:</source>
        <translation>Questa esecuzione ha usato {n} crediti. Cosa di solito aiuta:</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Check the word is English and singular (building, not batiments)</source>
        <translation>Verifica che la parola sia in inglese e al singolare (building, non batiments)</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Draw an example of one object (step 2)</source>
        <translation>Disegna un esempio di un oggetto (passaggio 2)</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Raise Detail so small objects are visible</source>
        <translation>Aumenta il Dettaglio per rendere visibili gli oggetti piccoli</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Try a smaller or different zone</source>
        <translation>Prova una zona più piccola o diversa</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detecting "{obj}"... · {n} found so far</source>
        <translation>Rilevamento di "{obj}"... · {n} trovati finora</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>How many objects sit at each confidence level.</source>
        <translation>Quanti oggetti si trovano a ciascun livello di confidenza.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Your 500 free detections are used up</source>
        <translation>I tuoi 500 rilevamenti gratuiti sono esauriti</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>10,000 detections every month (~1,700 km2)</source>
        <translation>10.000 rilevamenti al mese (~1.700 km2)</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Every building, tree, or road as clean polygons</source>
        <translation>Ogni edificio, albero o strada come poligoni netti</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Cancel anytime; your exported layers stay yours</source>
        <translation>Annulla in qualsiasi momento; i livelli esportati restano tuoi</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Opens your TerraLab dashboard</source>
        <translation>Apre la tua dashboard TerraLab</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Start Automatic AI Segmentation</source>
        <translation>Avvia AI Segmentation automatica</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Draw your zone</source>
        <translation>Disegna la tua zona</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Click on the map to outline the area to scan.</source>
        <translation>Clicca sulla mappa per delimitare l'area da analizzare.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Keep clicking around the area, at least 3 points.</source>
        <translation>Continua a cliccare intorno all'area, almeno 3 punti.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Click the first point to close the zone.</source>
        <translation>Clicca sul primo punto per chiudere la zona.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>undo point</source>
        <translation>annulla punto</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>cancel</source>
        <translation>annulla</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Now outline one object, then click the first point to close.</source>
        <translation>Ora delimita un oggetto, poi clicca sul primo punto per chiudere.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Click on the map to drop points around the area you want to scan.</source>
        <translation>Clicca sulla mappa per posizionare punti intorno all'area che vuoi analizzare.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Exit</source>
        <translation>Esci</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>e.g. building, tree, road, car</source>
        <translation>es. building, tree, road, car</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Browse ready-to-use objects with before / after previews.</source>
        <translation>Esplora oggetti pronti all'uso con anteprime prima / dopo.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Draw an example</source>
        <translation>Disegna un esempio</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Subscribe to finish this zone: 10,000 credits/month.</source>
        <translation>Abbonati per completare questa zona: 10.000 crediti/mese.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Blue = detected object</source>
        <translation>Blu = oggetto rilevato</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Filter detections by confidence. Lower shows more (and noisier), higher keeps only the strongest. Free and instant.</source>
        <translation>Filtra i rilevamenti per confidenza. Un valore più basso ne mostra di più (e più rumore), uno più alto conserva solo i più forti. Gratuito e istantaneo.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Show tiles (debug)</source>
        <translation>Mostra tile (debug)</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Open these detections in Manual mode to fix specific objects point-by-point, then come back and export.</source>
        <translation>Apri questi rilevamenti in modalità Manuale per correggere oggetti specifici punto per punto, poi torna qui per l'export.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Try instead:</source>
        <translation>Prova invece:</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>"{word}" will run as "{token}".</source>
        <translation>"{word}" verrà eseguito come "{token}".</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>That word isn't recognized - try a common object like building or tree.</source>
        <translation>Questa parola non è riconosciuta - prova un oggetto comune come building o tree.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>One object per run - start with the first one, then run again.</source>
        <translation>Un solo oggetto per esecuzione - inizia con il primo, poi rilancia.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>The Library has ready-to-use objects.</source>
        <translation>La Libreria contiene oggetti pronti all'uso.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Keep clicking to add points ({n} so far, 3 minimum).</source>
        <translation>Continua a cliccare per aggiungere punti ({n} finora, 3 minimo).</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>{n} points. Double-click or press Enter to finish, or click the first point to close.</source>
        <translation>{n} punti. Fai doppio clic o premi Invio per terminare, oppure clicca sul primo punto per chiudere.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>~ {credits} credits</source>
        <translation>~ {credits} crediti</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>{remaining} left</source>
        <translation>{remaining} rimanenti</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>{remaining} free left</source>
        <translation>{remaining} gratuiti rimanenti</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>~ {credits} credits · </source>
        <translation>~ {credits} crediti · </translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>will stop after {remaining}</source>
        <translation>si interromperà dopo {remaining}</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>1 credit ~ 0.17 km2 at default detail.</source>
        <translation>1 credito ~ 0,17 km2 al dettaglio predefinito.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Tip: draw an example of one {object} to catch more of them.</source>
        <translation>Suggerimento: disegna un esempio di un {object} per trovarne di più.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>object</source>
        <translation>oggetto</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Dense area: raise Detail to catch every object.</source>
        <translation>Area densa: aumenta il Dettaglio per catturare ogni oggetto.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/cards.py" line="0"/>
        <source>Restore</source>
        <translation>Ripristina</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/cards.py" line="0"/>
        <source>Delete forever</source>
        <translation>Elimina definitivamente</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/cards.py" line="0"/>
        <source>Deleted {when} · purges in {n} days</source>
        <translation>Eliminato {when} · verrà rimosso in {n} giorni</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/cards.py" line="0"/>
        <source>{tiles} tiles · {objects} objects · {credits} credits</source>
        <translation>{tiles} tile · {objects} oggetti · {credits} crediti</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/cards.py" line="0"/>
        <source>Older detection</source>
        <translation>Rilevamento più vecchio</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Details</source>
        <translation>Dettagli</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Fullscreen</source>
        <translation>Schermo intero</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Exit fullscreen</source>
        <translation>Esci da schermo intero</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Prompt</source>
        <translation>Prompt</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Copy</source>
        <translation>Copia</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Copy prompt</source>
        <translation>Copia prompt</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Copied</source>
        <translation>Copiato</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Template</source>
        <translation>Modello</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Your detection</source>
        <translation>Il tuo rilevamento</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Open the Library from the Automatic page to use this.</source>
        <translation>Apri la Libreria dalla pagina Automatico per usarlo.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>DATE</source>
        <translation>DATA</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>OBJECTS</source>
        <translation>OGGETTI</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>CREDITS</source>
        <translation>CREDITI</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>TILES</source>
        <translation>TILE</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>RESOLUTION</source>
        <translation>RISOLUZIONE</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>EXAMPLE</source>
        <translation>ESEMPIO</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Used</source>
        <translation>Usato</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Restore to map</source>
        <translation>Ripristina sulla mappa</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Reopens this run's review at the same place. Free - no credits.</source>
        <translation>Riapre la revisione di questa esecuzione nello stesso punto. Gratuito - nessun credito.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Export...</source>
        <translation>Export...</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Delete</source>
        <translation>Elimina</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Drop this object back into the prompt box for a new detection.</source>
        <translation>Rimetti questo oggetto nel campo del prompt per un nuovo rilevamento.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Remove from favorites</source>
        <translation>Rimuovi dai preferiti</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Add to favorites</source>
        <translation>Aggiungi ai preferiti</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Format:</source>
        <translation>Formato:</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>GeoPackage keeps the embedded style; other formats are saved without a style.</source>
        <translation>GeoPackage conserva lo stile incorporato; gli altri formati vengono salvati senza stile.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Browse...</source>
        <translation>Esplora...</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Your detections</source>
        <translation>I tuoi rilevamenti</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Recent</source>
        <translation>Recenti</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Favorites</source>
        <translation>Preferiti</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Recently deleted</source>
        <translation>Eliminati di recente</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Templates</source>
        <translation>Modelli</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Load older runs</source>
        <translation>Carica esecuzioni più vecchie</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Nothing here yet. Your automatic detections will land here, ready to reuse, restore or export.</source>
        <translation>Ancora niente qui. I tuoi rilevamenti automatici arriveranno qui, pronti per essere riutilizzati, ripristinati o esportati con Export.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Star a detection to keep it here.</source>
        <translation>Aggiungi un rilevamento ai preferiti per conservarlo qui.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Deleted runs wait here for 30 days, then they are purged for good.</source>
        <translation>Le esecuzioni eliminate restano qui per 30 giorni, poi vengono rimosse definitivamente.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>This permanently removes the stored previews and masks. Exported layers are never touched.</source>
        <translation>Questo rimuove definitivamente le anteprime e le maschere memorizzate. I livelli esportati non vengono mai toccati.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Could not load this run's stored detections. Try again later.</source>
        <translation>Impossibile caricare i rilevamenti memorizzati di questa esecuzione. Riprova più tardi.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Nothing to export at this confidence. Lower it and try again.</source>
        <translation>Niente da esportare con questo livello di confidenza. Abbassalo e riprova.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>The export failed. Check the file path and try again.</source>
        <translation>L'export non è riuscito. Controlla il percorso del file e riprova.</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Exported {n} polygon(s).</source>
        <translation>{n} poligono(i) esportato(i).</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Add a point</source>
        <translation>Aggiungi un punto</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Add area</source>
        <translation>Aggiungi superficie</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>All detections kept. Go 'Back to review' to export.</source>
        <translation>Tutti i rilevamenti conservati. Vai su «Torna alla revisione» per l'export.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Arrow keys</source>
        <translation>Tasti freccia</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Automatic</source>
        <translation>Automatico</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Automatic - detect and review</source>
        <translation>Automatico - rileva e rivedi</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Automatic - draw your zone</source>
        <translation>Automatico - disegna la tua zona</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Cancel the drawing</source>
        <translation>Annulla il disegno</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Cancel the running detection, or exit the review</source>
        <translation>Annulla il rilevamento in corso, oppure esci dalla revisione</translation>
    </message>
    <message>
        <location filename="../src/core/feature_encoder.py" line="0"/>
        <source>Click</source>
        <translation>Clic</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_run.py" line="0"/>
        <source>Could not place the example on the image. Redraw the example box inside the zone and try again.</source>
        <translation>Impossibile posizionare l'esempio sull'immagine. Ridisegna il riquadro dell'esempio all'interno della zona e riprova.</translation>
    </message>
    <message>
        <location filename="../src/core/run_restore.py" line="0"/>
        <source>Could not rebuild this run's detections.</source>
        <translation>Impossibile ricostruire i rilevamenti di questa esecuzione.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Delete removes this object</source>
        <translation>Canc rimuove questo oggetto</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Delete the active object</source>
        <translation>Elimina l'oggetto attivo</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Detect objects, or export the reviewed polygons</source>
        <translation>Rileva oggetti, oppure esporta i poligoni revisionati</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detection continues in the background. Reopen AI Segmentation to follow it.</source>
        <translation>Il rilevamento continua in background. Riapri AI Segmentation per seguirlo.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Double-click</source>
        <translation>Doppio clic</translation>
    </message>
    <message>
        <location filename="../src/core/pip_diagnostics.py" line="0"/>
        <source>Example</source>
        <translation>Esempio</translation>
    </message>
    <message>
        <location filename="../src/core/run_restore.py" line="0"/>
        <source>Finish or exit the current run before restoring a past one.</source>
        <translation>Termina o esci dall'esecuzione in corso prima di ripristinarne una precedente.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Finish the zone</source>
        <translation>Chiudi la zona</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>General</source>
        <translation>Generale</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Hand-refined objects are always kept, whatever the confidence.</source>
        <translation>Gli oggetti ritoccati a mano vengono sempre conservati, indipendentemente dalla confidenza.</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/manual_handoff.py" line="0"/>
        <source>Install now</source>
        <translation>Installa ora</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Keeps this polygon in your session. Export writes all kept polygons to a layer.</source>
        <translation>Conserva questo poligono nella tua sessione. Export scrive tutti i poligoni conservati in un livello.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_maptool.py" line="0"/>
        <source>Keyboard shortcuts</source>
        <translation>Scorciatoie da tastiera</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Left-click</source>
        <translation>Clic sinistro</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Left-click adds area · Right-click removes area</source>
        <translation>Clic sinistro aggiunge superficie · Clic destro la rimuove</translation>
    </message>
    <message>
        <location filename="../src/core/layer_conventions.py" line="0"/>
        <source>Manual</source>
        <translation>Manuale</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/manual_handoff.py" line="0"/>
        <source>Manual mode needs a one-time setup</source>
        <translation>La modalità Manuale richiede una configurazione iniziale</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Manual session</source>
        <translation>Sessione manuale</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Max detail for this zone - draw a larger zone for finer detail.</source>
        <translation>Dettaglio massimo per questa zona - disegna una zona più grande per un dettaglio più fine.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Merges your edits back into the review. Nothing is exported yet.</source>
        <translation>Reintegra le tue modifiche nella revisione. Non è ancora stato esportato nulla.</translation>
    </message>
    <message>
        <location filename="../src/core/venv_manager.py" line="0"/>
        <source>OK</source>
        <translation>OK</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>One color per object - check neighbors are separated</source>
        <translation>Un colore per oggetto - verifica che gli oggetti vicini siano separati</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Optional shape and size controls: simplify outlines, clean edges, round corners, expand or shrink, fill holes, size filters.</source>
        <translation>Controlli opzionali di forma e dimensione: semplifica contorni, pulisci bordi, arrotonda angoli, dilata o contrai, riempi buchi, filtri di dimensione.</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_lifecycle.py" line="0"/>
        <source>Out of credits after {done}/{total} tiles. Your detections are kept below.</source>
        <translation>Crediti esauriti dopo {done}/{total} tile. I tuoi rilevamenti sono conservati qui sotto.</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_results.py" line="0"/>
        <source>Outlines only - check boundaries against the imagery</source>
        <translation>Solo contorni - verifica i confini rispetto alle immagini</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Pan the map</source>
        <translation>Scorri la mappa</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Part of your zone is outside "{layer}" - only the overlapping area will return objects.</source>
        <translation>Parte della tua zona è fuori da "{layer}" - solo l'area sovrapposta restituirà oggetti.</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_run.py" line="0"/>
        <source>Pick an object to detect first (nothing was selected).</source>
        <translation>Scegli prima un oggetto da rilevare (nulla era selezionato).</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Polygon saved ({n} total). Click another element, or export when done.</source>
        <translation>Poligono salvato ({n} totali). Clicca su un altro elemento, oppure fai l'export quando hai finito.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Press S to keep it (turns green)</source>
        <translation>Premi S per conservarlo (diventa verde)</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Refine detections</source>
        <translation>Affina rilevamenti</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_results.py" line="0"/>
        <source>Refine seeds</source>
        <translation>Rilevamenti da affinare</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/manual_handoff.py" line="0"/>
        <source>Refining uses the free local AI, which is not installed yet. Install it now (a few minutes, automatic)? Your detections stay safely in the review.</source>
        <translation>L'affinamento usa l'IA locale gratuita, che non è ancora installata. Installarla ora (pochi minuti, automatico)? I tuoi rilevamenti restano al sicuro nella revisione.</translation>
    </message>
    <message>
        <location filename="../src/core/checkpoint_manager.py" line="0"/>
        <source>Remove</source>
        <translation>Rimuovi</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Remove area</source>
        <translation>Rimuovi superficie</translation>
    </message>
    <message>
        <location filename="../src/ui/zone_selection_maptool.py" line="0"/>
        <source>Remove zone</source>
        <translation>Rimuovi zona</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Resets {date}</source>
        <translation>Si rinnova il {date}</translation>
    </message>
    <message>
        <location filename="../src/core/run_restore.py" line="0"/>
        <source>Restored "{prompt}" - adjust and export below.</source>
        <translation>"{prompt}" ripristinato - modifica ed esporta qui sotto.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Right-click</source>
        <translation>Clic destro</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Save polygon (S)</source>
        <translation>Salva poligono (S)</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_lifecycle.py" line="0"/>
        <source>Session expired. Sign in again to continue.</source>
        <translation>Sessione scaduta. Accedi di nuovo per continuare.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Start (the visible mode's Start button)</source>
        <translation>Avvia (il pulsante Avvia della modalità visibile)</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Start Manual AI Segmentation</source>
        <translation>Avvia AI Segmentation manuale</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_lifecycle.py" line="0"/>
        <source>The raster was removed. Your polygons were saved to a layer.</source>
        <translation>Il raster è stato rimosso. I tuoi poligoni sono stati salvati in un livello.</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_lifecycle.py" line="0"/>
        <source>The selected raster was removed.</source>
        <translation>Il raster selezionato è stato rimosso.</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>The selected raster was removed. Keeping what was already found.</source>
        <translation>Il raster selezionato è stato rimosso. Ciò che era già stato trovato viene conservato.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Tip: S saves, Enter exports, Ctrl+Z undoes a click.</source>
        <translation>Suggerimento: S salva, Invio esporta, Ctrl+Z annulla un clic.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Undo the last point</source>
        <translation>Annulla l'ultimo punto</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Writes a GeoPackage layer with your {n} kept polygons.</source>
        <translation>Scrive un livello GeoPackage con i tuoi {n} poligoni conservati.</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_results.py" line="0"/>
        <source>Yellow = confident · Purple = uncertain</source>
        <translation>Giallo = confidente · Viola = incerto</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Your zone is outside "{layer}". Pick the right layer or draw inside it.</source>
        <translation>La tua zona è fuori da "{layer}". Scegli il livello corretto o disegna al suo interno.</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_zone.py" line="0"/>
        <source>Zone too large. Reduce the area to {max} tiles or fewer.</source>
        <translation>Zona troppo grande. Riduci l'area a {max} tile o meno.</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>objects</source>
        <translation>oggetti</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{kept} of {total} detections kept - click a blue detection to edit it</source>
        <translation>{kept} rilevamenti conservati su {total} - clicca su un rilevamento blu per modificarlo</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} of {total} free detections left</source>
        <translation>{n} rilevamenti gratuiti rimanenti su {total}</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{remaining} free detections left.</source>
        <translation>{remaining} rilevamenti gratuiti rimanenti.</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>≈ 1 tile = 1 credit</source>
        <translation>≈ 1 tile = 1 credito</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Keep this result</source>
        <translation>Conserva questo risultato</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Start over</source>
        <translation>Ricomincia</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Adjust and run again</source>
        <translation>Modifica e rilancia</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>all shown</source>
        <translation>tutti mostrati</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>{visible} of {n} shown</source>
        <translation>{visible} su {n} mostrati</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>{hidden} below {pct}%</source>
        <translation>{hidden} sotto {pct}%</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Adjust and run again</source>
        <translation>Modifica e rilancia</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Keep this result</source>
        <translation>Conserva questo risultato</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Start over</source>
        <translation>Ricomincia</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>all shown</source>
        <translation>tutti mostrati</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>{hidden} below {pct}%</source>
        <translation>{hidden} sotto {pct}%</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>{visible} of {n} shown</source>
        <translation>{visible} su {n} mostrati</translation>
    </message>
    <message>
        <source>Right angles:</source>
        <translation>Angoli retti:</translation>
    </message>
    <message>
        <source>Snap edges to 90 degrees for man-made shapes like buildings, pools and solar panels.</source>
        <translation>Allinea i bordi a 90 gradi per forme artificiali come edifici, piscine e pannelli solari.</translation>
    </message>
    <message>
        <source>Click an object on the map and the AI outlines it. You go one object at a time, checking and saving each polygon yourself.</source>
        <translation>Clicca su un oggetto sulla mappa e l'IA lo delimita. Procedi un oggetto alla volta, verificando e salvando ogni poligono tu stesso.</translation>
    </message>
    <!-- Onboarding conversion batch (2026-07-04) -->
    <message>
        <source>Show guidance tips again</source>
        <translation>Mostra di nuovo i suggerimenti</translation>
    </message>
    <message>
        <source>Guidance tips restored</source>
        <translation>Suggerimenti ripristinati</translation>
    </message>
    <message>
        <source>Run again here</source>
        <translation>Rilancia qui</translation>
    </message>
    <message>
        <source>Reload this zone and object, ready to detect.</source>
        <translation>Ricarica questa zona e questo oggetto, pronti per rilevare.</translation>
    </message>
    <message>
        <source>Same object, new zone</source>
        <translation>Stesso oggetto, nuova zona</translation>
    </message>
    <message>
        <source>Keep this object and draw a new zone on the map.</source>
        <translation>Conserva questo oggetto e disegna una nuova zona sulla mappa.</translation>
    </message>
    <message>
        <source>Upgrade to Pro</source>
        <translation>Passa a Pro</translation>
    </message>
    <message>
        <source>Free account - sign up takes 15 seconds in your browser.</source>
        <translation>Account gratuito - la registrazione richiede 15 secondi nel tuo browser.</translation>
    </message>
    <message>
        <source>Manual mode stays free and unlimited on your computer.</source>
        <translation>La modalità Manuale resta gratuita e illimitata sul tuo computer.</translation>
    </message>
    <message>
        <source>Finds every object of one kind in your zone - draw a zone, name the object, get all the polygons at once.</source>
        <translation>Trova tutti gli oggetti di un tipo nella tua zona - disegna una zona, nomina l'oggetto, ottieni tutti i poligoni in una volta.</translation>
    </message>
    <message>
        <source>Tip: draw an example on the map to boost detection of unusual objects.</source>
        <translation>Suggerimento: disegna un esempio sulla mappa per migliorare il rilevamento di oggetti inusuali.</translation>
    </message>
    <message>
        <source>Tip: lower Confidence to reveal more detections, raise it to keep only the best.</source>
        <translation>Suggerimento: abbassa la Confidenza per rivelare più rilevamenti, aumentala per conservare solo i migliori.</translation>
    </message>
    <message>
        <source>This zone is {area} km2 - free trial zones go up to {max} km2.</source>
        <translation>Questa zona è di {area} km2 - le zone della prova gratuita arrivano fino a {max} km2.</translation>
    </message>
    <message>
        <source>Draw a smaller zone, or &lt;a href=&quot;{url}&quot;&gt;subscribe&lt;/a&gt; to segment areas of any size.</source>
        <translation>Disegna una zona più piccola, oppure &lt;a href=&quot;{url}&quot;&gt;abbonati&lt;/a&gt; per segmentare aree di qualsiasi dimensione.</translation>
    </message>
    <message>
        <source>Running low: {n} free detections left. &lt;a href=&quot;{url}&quot;&gt;Subscribe&lt;/a&gt; to keep going.</source>
        <translation>In esaurimento: {n} rilevamenti gratuiti rimanenti. &lt;a href=&quot;{url}&quot;&gt;Abbonati&lt;/a&gt; per continuare.</translation>
    </message>
    <message>
        <source>Last run: {count} {object} exported · {area} km2 · {used} credits used, {left} left</source>
        <translation>Ultima esecuzione: {count} {object} esportato(i) · {area} km2 · {used} crediti usati, {left} rimanenti</translation>
    </message>
    <message>
        <source>Last run: {count} {object} exported · {area} km2 · {used} credits used</source>
        <translation>Ultima esecuzione: {count} {object} esportato(i) · {area} km2 · {used} crediti usati</translation>
    </message>
    <message>
        <source>Then segment any imagery: point and click, or fully automatic.</source>
        <translation>Poi segmenta qualsiasi immagine: punta e clicca, oppure in modo completamente automatico.</translation>
    </message>
    <message>
        <source>Waiting for your browser sign-in...</source>
        <translation>In attesa dell'accesso nel browser...</translation>
    </message>
    <message>
        <source>New: Automatic mode finds every object in a zone at once.</source>
        <translation>Novità: la modalità Automatica trova tutti gli oggetti di una zona in una volta.</translation>
    </message>
    <message>
        <source>Try Automatic</source>
        <translation>Prova Automatico</translation>
    </message>
    <message>
        <source>Got it - hide this tip</source>
        <translation>Capito - nascondi questo suggerimento</translation>
    </message>
    <message>
        <source>Finish or cancel the current detection before re-running a past one.</source>
        <translation>Termina o annulla il rilevamento in corso prima di rilanciarne uno precedente.</translation>
    </message>
    <message>
        <source>Couldn&apos;t open your browser. Check your connection and click Sign in / Sign up to start again.</source>
        <translation>Impossibile aprire il tuo browser. Controlla la connessione e clicca su Accedi / Registrati per ricominciare.</translation>
    </message>
</context>
</TS>
