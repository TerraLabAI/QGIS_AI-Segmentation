<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="zh_TW">
<context>
    <name>AISegmentation</name>
    <message>
        <source>Display colors:</source>
        <translation>顯示顏色：</translation>
    </message>
    <message>
        <source>Normal</source>
        <translation>正常</translation>
    </message>
    <message>
        <source>Confidence</source>
        <translation>信心度</translation>
    </message>
    <message>
        <source>Random</source>
        <translation>隨機</translation>
    </message>
    <message>
        <source>Outline</source>
        <translation>輪廓</translation>
    </message>
    <message>
        <source>How detections are coloured on the map (visual only): Normal outline, Confidence heatmap (green sure, red uncertain), or a random colour per object to tell them apart.</source>
        <translation>偵測結果在地圖上的顯色方式（僅影響外觀）：一般輪廓、信心度熱區圖（綠色代表確定、紅色代表不確定），或依物件顯示隨機顏色以便區分。</translation>
    </message>
    <message>
        <source>Retry</source>
        <translation>重試</translation>
    </message>
    <message>
        <source>Go back to your zone, references and settings to adjust and detect again. Nothing is saved.</source>
        <translation>回到您的範圍、參考影像和設定進行調整，再重新偵測。系統不會儲存任何內容。</translation>
    </message>
    <message>
        <source>Click to enlarge</source>
        <translation>點擊放大</translation>
    </message>
    <message>
        <source>This is exactly what the AI uses: your object with a little of its surroundings.</source>
        <translation>這正是 AI 所使用的內容：您的物件加上周圍一小部分環境。</translation>
    </message>
    <message>
        <source>Describe what to find</source>
        <translation>描述要尋找的內容</translation>
    </message>
    <message>
        <source>solar panel, building, tree…</source>
        <translation>solar panel, building, tree…</translation>
    </message>
    <message>
        <source>1-2 words, English</source>
        <translation>1-2 個字，英文</translation>
    </message>
    <message>
        <source>optional</source>
        <translation>選填</translation>
    </message>
    <message>
        <source>Show an example</source>
        <translation>顯示範例</translation>
    </message>
    <message>
        <source>Outline one object; the AI finds the rest. No good name for it? Examples alone work too.</source>
        <translation>圈選一個物件，AI 會找出其他相似物件。想不到合適的名稱？只用範例也可以。</translation>
    </message>
    <message>
        <source>Exclude a look-alike</source>
        <translation>排除相似物件</translation>
    </message>
    <message>
        <source>Now outline a look-alike to exclude, then click the first point to close.</source>
        <translation>現在圈選一個要排除的相似物件，再點擊起點以封閉範圍。</translation>
    </message>
    <message>
        <source>Your examples drive the search.</source>
        <translation>您的範例將引導搜尋。</translation>
    </message>
    <message>
        <source>Too generic to name. Clear the box to search from your example alone, or type a concrete object.</source>
        <translation>名稱太籠統。清空欄位僅以範例搜尋，或輸入具體的物件名稱。</translation>
    </message>
    <message>
        <source>Example match</source>
        <translation>範例比對</translation>
    </message>
    <message>
        <source>Include</source>
        <translation>納入</translation>
    </message>
    <message>
        <source>Exclude</source>
        <translation>排除</translation>
    </message>
    <message>
        <source>Mark an object to find more like it.</source>
        <translation>標記一個物件以尋找更多類似的物件。</translation>
    </message>
    <message>
        <source>Mark a false positive to drop things like it.</source>
        <translation>標記一個誤判物件，以剔除類似的結果。</translation>
    </message>
    <message>
        <source>Draw on map</source>
        <translation>在地圖上繪製</translation>
    </message>
    <message>
        <source>Outline one object on the map; SAM finds all similar ones.</source>
        <translation>在地圖上圈選一個物件，SAM 會找出所有相似物件。</translation>
    </message>
    <message>
        <source>Finer detail finds smaller objects.</source>
        <translation>更精細的細節可偵測更小的物件。</translation>
    </message>
    <message>
        <source>{n} object(s) detected</source>
        <translation>偵測到 {n} 個物件</translation>
    </message>
    <message>
        <source>Adjust below, then export</source>
        <translation>在下方調整，然後 Export</translation>
    </message>
    <message>
        <source>Refine in Manual mode</source>
        <translation>在手動模式中細修</translation>
    </message>
    <message>
        <source>Some objects off? Refine them in Manual mode first.</source>
        <translation>有些物件不太準確？請先在手動模式中細修。</translation>
    </message>
    <message>
        <source>Now outline one object on the map, then double-click to finish.</source>
        <translation>現在在地圖上圈選一個物件，再雙擊以完成。</translation>
    </message>
    <message>
        <source>Now outline one false positive on the map, then double-click to finish.</source>
        <translation>現在在地圖上圈選一個誤判物件，再雙擊以完成。</translation>
    </message>
    <message>
        <source>Refine in Manual</source>
        <translation>在手動模式中細修</translation>
    </message>
    <message>
        <source>Open these detections in Manual mode to fix specific objects with point-and-click, then return here to Finish.</source>
        <translation>在手動模式中開啟這些偵測結果，以點擊方式修正特定物件，完成後返回此處按「完成」。</translation>
    </message>
    <message>
        <source>Refining Automatic results</source>
        <translation>正在細修自動偵測結果</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Fine-tune the detections, then go back to review to export.</source>
        <translation>微調偵測結果，再返回檢視畫面進行 Export。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Editing this detection.</source>
        <translation>正在編輯此偵測結果。</translation>
    </message>
    <message>
        <source>Editing this detection</source>
        <translation>正在編輯此偵測結果</translation>
    </message>
    <message>
        <source>adds area</source>
        <translation>新增區域</translation>
    </message>
    <message>
        <source>removes area</source>
        <translation>移除區域</translation>
    </message>
    <message>
        <source>keeps it (turns green)</source>
        <translation>保留它（會變成綠色）</translation>
    </message>
    <message>
        <source>removes the object</source>
        <translation>移除該物件</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Press S to keep it (turns green) · Delete removes it</source>
        <translation>按 S 保留它（會變成綠色）· 按 Delete 移除它</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Click a blue detection to open it for editing.</source>
        <translation>點擊藍色的偵測結果以開啟編輯。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Left-click adds area, right-click removes it. Press S to keep it (turns green).</source>
        <translation>左鍵點擊新增區域，右鍵點擊移除區域。按 S 保留它（會變成綠色）。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>{kept} of {total} kept - 'Back to review' to export.</source>
        <translation>已保留 {kept}／{total} 個－點擊「返回檢視」進行 Export。</translation>
    </message>
    <message>
        <source>Back to review</source>
        <translation>返回檢視</translation>
    </message>
    <message>
        <source>Finish or go back to review to switch modes.</source>
        <translation>請先完成或返回檢視，才能切換模式。</translation>
    </message>
    <message>
        <source>Finish or exit the review to switch modes.</source>
        <translation>請先完成或退出檢視，才能切換模式。</translation>
    </message>
    <message>
        <source>Preparing Manual mode, loading the local model...</source>
        <translation>正在準備手動模式，載入本機模型...</translation>
    </message>
    <message>
        <source>Blue = detections to review, one at a time.</source>
        <translation>藍色＝待檢視的偵測結果，一次處理一個。</translation>
    </message>
    <message>
        <source>Left-click a detection to edit it (adds area); right-click to remove a part</source>
        <translation>左鍵點擊偵測結果以編輯（新增區域）；右鍵點擊以移除部分區域</translation>
    </message>
    <message>
        <source>Press S to validate it (turns green), then move on to the next one.</source>
        <translation>按 S 確認它（會變成綠色），然後繼續下一個。</translation>
    </message>
    <message>
        <source>Locked - refined in Manual mode</source>
        <translation>已鎖定－已在手動模式中細修</translation>
    </message>
    <message>
        <source>Confidence is locked while you refine in Manual mode.</source>
        <translation>在手動模式中細修時，信心度會鎖定。</translation>
    </message>
    <message>
        <source>Refining in Manual needs the local model. Open Manual mode once to finish setup, then try again.</source>
        <translation>在手動模式中細修需要本機模型。請先開啟手動模式完成設定，再試一次。</translation>
    </message>
    <message>
        <source>Detection</source>
        <translation>偵測結果</translation>
    </message>
    <message>
        <source>Confidence:</source>
        <translation>信心度：</translation>
    </message>
    <message>
        <source>Minimum confidence to keep a detected object. Lower finds more objects but may add false positives; raise it for cleaner results on large, distinct features.</source>
        <translation>保留偵測物件所需的最低信心度。數值越低可偵測到更多物件，但可能增加誤判；數值越高則能在大型、明顯的地物上取得更乾淨的結果。</translation>
    </message>
    <message>
        <source>Dependencies</source>
        <translation>相依套件</translation>
    </message>
    <message>
        <source>Local AI model files stored on this computer.</source>
        <translation>儲存在這台電腦上的本機 AI 模型檔案。</translation>
    </message>
    <message>
        <source>On disk</source>
        <translation>佔用空間</translation>
    </message>
    <message>
        <source>Not installed</source>
        <translation>未安裝</translation>
    </message>
    <message>
        <source>Open folder</source>
        <translation>開啟資料夾</translation>
    </message>
    <message>
        <source>Accept the Terms and Privacy Policy to enable segmentation.</source>
        <translation>請接受服務條款與隱私政策以啟用分割功能。</translation>
    </message>
    <message>
        <source>An unexpected error occurred during export. Please check the logs.</source>
        <translation>Export 時發生未預期的錯誤，請查看日誌。</translation>
    </message>
    <message>
        <source>I agree to the &lt;a href=&quot;{terms}&quot;&gt;Terms&lt;/a&gt; and &lt;a href=&quot;{privacy}&quot;&gt;Privacy Policy&lt;/a&gt;</source>
        <translation>我同意&lt;a href=&quot;{terms}&quot;&gt;服務條款&lt;/a&gt;與&lt;a href=&quot;{privacy}&quot;&gt;隱私政策&lt;/a&gt;</translation>
    </message>
    <message>
        <source>No valid polygons could be created from the selection. Try adjusting the refine settings or making a new selection.</source>
        <translation>無法從此選取範圍建立有效的多邊形。請試著調整細修設定或重新選取。</translation>
    </message>
    <message>
        <source>Repairing Installation</source>
        <translation>正在修復安裝</translation>
    </message>
    <message>
        <source>Repairing installation...</source>
        <translation>正在修復安裝...</translation>
    </message>
    <message>
        <source>Restart QGIS Required</source>
        <translation>需要重新啟動 QGIS</translation>
    </message>
    <message>
        <source>Something went wrong with this click, so it was not applied. Please try again.</source>
        <translation>此次點擊發生問題，未套用變更，請再試一次。</translation>
    </message>
    <message>
        <source>The Python runtime used by the AI engine is damaged (this can be caused by a disk cleanup tool or antivirus). It will now be repaired automatically. Please try your selection again once the repair finishes.</source>
        <translation>AI 引擎所使用的 Python 執行環境已損毀（可能是磁碟清理工具或防毒軟體所造成）。系統現在會自動修復，修復完成後請重新選取一次。</translation>
    </message>
    <message>
        <source>Your polygons were added as a temporary layer so nothing is lost.</source>
        <translation>您的多邊形已加入暫時圖層，不會遺失任何內容。</translation>
    </message>
    <message>
        <source>Could not write to {name}. Saved to a separate file instead.</source>
        <translation>無法寫入 {name}，已改存為另一個檔案。</translation>
    </message>
    <message>
        <source>Click Install to set up AI Segmentation</source>
        <translation>點擊「安裝」以設定 AI Segmentation</translation>
    </message>
    <message>
        <source>Installing AI Segmentation...</source>
        <translation>正在安裝 AI Segmentation...</translation>
    </message>
    <message>
        <source>Verifying installation...</source>
        <translation>正在驗證安裝...</translation>
    </message>
    <message>
        <source>Detecting device...</source>
        <translation>正在偵測裝置...</translation>
    </message>
    <message>
        <source>Install path: {}</source>
        <translation>安裝路徑：{}</translation>
    </message>
    <message>
        <source>To install in a different folder, set the environment variable AI_SEGMENTATION_CACHE_DIR:</source>
        <translation>若要安裝到其他資料夾，請設定環境變數 AI_SEGMENTATION_CACHE_DIR：</translation>
    </message>
    <message>
        <source>1. Open Windows Settings &gt; System &gt; Advanced system settings
2. Click 'Environment Variables'
3. Under 'User variables', click 'New'
4. Variable name: AI_SEGMENTATION_CACHE_DIR
5. Variable value: the folder path you want to use
6. Click OK and restart QGIS</source>
        <translation>1. 開啟 Windows 設定 &gt; 系統 &gt; 進階系統設定
2. 點擊「環境變數」
3. 在「使用者變數」下，點擊「新增」
4. 變數名稱：AI_SEGMENTATION_CACHE_DIR
5. 變數值：您想使用的資料夾路徑
6. 點擊確定並重新啟動 QGIS</translation>
    </message>
    <message>
        <source>Run this command in Terminal, then restart QGIS:

launchctl setenv AI_SEGMENTATION_CACHE_DIR /your/path</source>
        <translation>請在終端機執行以下指令，然後重新啟動 QGIS：

launchctl setenv AI_SEGMENTATION_CACHE_DIR /your/path</translation>
    </message>
    <message>
        <source>Add this line to your ~/.bashrc or ~/.profile, then restart QGIS:

export AI_SEGMENTATION_CACHE_DIR=/your/path</source>
        <translation>請將以下這行加入您的 ~/.bashrc 或 ~/.profile，然後重新啟動 QGIS：

export AI_SEGMENTATION_CACHE_DIR=/your/path</translation>
    </message>
    <message>
        <source>Checking...</source>
        <translation>正在檢查...</translation>
    </message>
    <message>
        <source>Install</source>
        <translation>安裝</translation>
    </message>
    <message>
        <source>Update</source>
        <translation>更新</translation>
    </message>
    <message>
        <source>Dependencies ready</source>
        <translation>相依套件已就緒</translation>
    </message>
    <message>
        <source>Downloading AI model...</source>
        <translation>正在下載 AI 模型...</translation>
    </message>
    <message>
        <source>Dependencies ready, model not downloaded</source>
        <translation>相依套件已就緒，模型尚未下載</translation>
    </message>
    <message>
        <source>Dependencies ready, model download failed</source>
        <translation>相依套件已就緒，模型下載失敗</translation>
    </message>
    <message>
        <source>Download Model</source>
        <translation>下載模型</translation>
    </message>
    <message>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <source>Cancel installation</source>
        <translation>取消安裝</translation>
    </message>
    <message>
        <source>Are you sure you want to cancel the installation?</source>
        <translation>您確定要取消安裝嗎？</translation>
    </message>
    <message>
        <source>Installation cancelled</source>
        <translation>安裝已取消</translation>
    </message>
    <message>
        <source>Installation failed</source>
        <translation>安裝失敗</translation>
    </message>
    <message>
        <source>Verification failed:</source>
        <translation>驗證失敗：</translation>
    </message>
    <message>
        <source>Verification Failed</source>
        <translation>驗證失敗</translation>
    </message>
    <message>
        <source>Virtual environment was created but verification failed:</source>
        <translation>已建立虛擬環境，但驗證失敗：</translation>
    </message>
    <message>
        <source>Unknown error</source>
        <translation>未知錯誤</translation>
    </message>
    <message>
        <source>Installation Failed</source>
        <translation>安裝失敗</translation>
    </message>
    <message>
        <source>Update QGIS to 3.34+ for the latest AI model</source>
        <translation>將 QGIS 更新至 3.34 以上版本以使用最新的 AI 模型</translation>
    </message>
    <message>
        <source>Intel Mac: using SAM1 (compatible with PyTorch 2.2)</source>
        <translation>Intel Mac：使用 SAM1（相容於 PyTorch 2.2）</translation>
    </message>
    <message>
        <source>Download Failed</source>
        <translation>下載失敗</translation>
    </message>
    <message>
        <source>Failed to download model:</source>
        <translation>模型下載失敗：</translation>
    </message>
    <message>
        <source>AI Segmentation by TerraLab</source>
        <translation>AI Segmentation by TerraLab</translation>
    </message>
    <message>
        <source>Select a Raster Layer to Segment:</source>
        <translation>選擇要分割的點陣圖圖層：</translation>
    </message>
    <message>
        <source>Select a raster layer (GeoTIFF, WMS, XYZ tiles, etc.)</source>
        <translation>選擇點陣圖圖層（GeoTIFF、WMS、XYZ 圖磚等）</translation>
    </message>
    <message>
        <source>No raster layer found. Add a GeoTIFF, image file, or online layer (WMS, XYZ) to your project.</source>
        <translation>找不到點陣圖圖層。請將 GeoTIFF、影像檔或線上圖層（WMS、XYZ）加入您的專案。</translation>
    </message>
    <message>
        <source>No layer found. Add a raster or online layer to your project.</source>
        <translation>找不到圖層。請將點陣圖或線上圖層加入您的專案。</translation>
    </message>
    <message>
        <source>Start AI Segmentation</source>
        <translation>開始 AI Segmentation</translation>
    </message>
    <message>
        <source>Save polygon</source>
        <translation>儲存多邊形</translation>
    </message>
    <message>
        <source>Undo last point</source>
        <translation>復原上一個點</translation>
    </message>
    <message>
        <source>Stop segmentation</source>
        <translation>停止分割</translation>
    </message>
    <message>
        <source>Segmentation</source>
        <translation>分割</translation>
    </message>
    <message>
        <source>Navigation</source>
        <translation>導覽</translation>
    </message>
    <message>
        <source>Space</source>
        <translation>空格鍵</translation>
    </message>
    <message>
        <source>Hold and move to pan the map</source>
        <translation>按住並移動以平移地圖</translation>
    </message>
    <message>
        <source>Middle mouse button</source>
        <translation>滑鼠中鍵</translation>
    </message>
    <message>
        <source>Click and drag to pan the map</source>
        <translation>點擊並拖曳以平移地圖</translation>
    </message>
    <message>
        <source>Shortcuts</source>
        <translation>快速鍵</translation>
    </message>
    <message>
        <source>Save current polygon to your session</source>
        <translation>將目前的多邊形儲存至此工作階段</translation>
    </message>
    <message>
        <source>The AI model works best on one element at a time.</source>
        <translation>AI 模型在一次處理一個地物時效果最佳。</translation>
    </message>
    <message>
        <source>Save your polygon before selecting the next element.</source>
        <translation>選取下一個地物之前，請先儲存您的多邊形。</translation>
    </message>
    <message>
        <source>Export polygon to a layer</source>
        <translation>Export 多邊形至圖層</translation>
    </message>
    <message>
        <source>Export {count} polygons to a layer</source>
        <translation>Export {count} 個多邊形至圖層</translation>
    </message>
    <message>
        <source>Refine selection</source>
        <translation>細修選取範圍</translation>
    </message>
    <message>
        <source>Expand/Contract:</source>
        <translation>擴張／收縮：</translation>
    </message>
    <message>
        <source>Positive = expand outward, Negative = shrink inward</source>
        <translation>正值＝向外擴張，負值＝向內收縮</translation>
    </message>
    <message>
        <source>Simplify outline:</source>
        <translation>簡化輪廓：</translation>
    </message>
    <message>
        <source>Reduce small variations in the outline (0 = no change)</source>
        <translation>減少輪廓上的細微變化（0＝不變更）</translation>
    </message>
    <message>
        <source>Fill holes:</source>
        <translation>填補孔洞：</translation>
    </message>
    <message>
        <source>Fill interior holes in the selection</source>
        <translation>填補選取範圍內部的孔洞</translation>
    </message>
    <message>
        <source>Min area:</source>
        <translation>最小面積：</translation>
    </message>
    <message>
        <source>Remove polygons smaller than this area (in pixels)</source>
        <translation>移除面積小於此值的多邊形（以像素計）</translation>
    </message>
    <message>
        <source>Shape</source>
        <translation>形狀</translation>
    </message>
    <message>
        <source>Size</source>
        <translation>大小</translation>
    </message>
    <message>
        <source>Round corners:</source>
        <translation>圓角：</translation>
    </message>
    <message>
        <source>Round corners for natural shapes like trees and bushes. Increase 'Simplify outline' for smoother results.</source>
        <translation>為樹木、灌木等自然形狀套用圓角。提高「簡化輪廓」可讓結果更平滑。</translation>
    </message>
    <message>
        <source>Outline</source>
        <translation>輪廓</translation>
    </message>
    <message>
        <source>Selection</source>
        <translation>選取範圍</translation>
    </message>
    <message>
        <source>Click on the element you want to segment:</source>
        <translation>點擊您要分割的地物：</translation>
    </message>
    <message>
        <source>Left-click to select</source>
        <translation>左鍵點擊以選取</translation>
    </message>
    <message>
        <source>Left-click to add more</source>
        <translation>左鍵點擊以新增更多</translation>
    </message>
    <message>
        <source>Right-click to exclude from selection</source>
        <translation>右鍵點擊以從選取範圍中排除</translation>
    </message>
    <message>
        <source>Invalid Layer</source>
        <translation>無效的圖層</translation>
    </message>
    <message>
        <source>Layer extent contains invalid coordinates (NaN/Inf). Check the raster file.</source>
        <translation>圖層範圍包含無效座標（NaN／Inf）。請檢查點陣圖檔案。</translation>
    </message>
    <message>
        <source>Not Ready</source>
        <translation>尚未就緒</translation>
    </message>
    <message>
        <source>Please wait for the SAM model to load.</source>
        <translation>請等待 SAM 模型載入完成。</translation>
    </message>
    <message>
        <source>Load Failed</source>
        <translation>載入失敗</translation>
    </message>
    <message>
        <source>Layer Creation Failed</source>
        <translation>建立圖層失敗</translation>
    </message>
    <message>
        <source>Could not create the output layer.</source>
        <translation>無法建立輸出圖層。</translation>
    </message>
    <message>
        <source>Export Failed</source>
        <translation>Export 失敗</translation>
    </message>
    <message>
        <source>Could not save layer to file:</source>
        <translation>無法將圖層儲存至檔案：</translation>
    </message>
    <message>
        <source>Layer was saved but could not be loaded:</source>
        <translation>圖層已儲存，但無法載入：</translation>
    </message>
    <message>
        <source>You have {count} unsaved polygon(s).</source>
        <translation>您有 {count} 個尚未儲存的多邊形。</translation>
    </message>
    <message>
        <source>Changing layer will discard your current segmentation. Continue?</source>
        <translation>變更圖層將捨棄目前的分割結果，是否繼續？</translation>
    </message>
    <message>
        <source>Change Layer?</source>
        <translation>變更圖層？</translation>
    </message>
    <message>
        <source>Stop Segmentation?</source>
        <translation>停止分割？</translation>
    </message>
    <message>
        <source>This will discard {count} polygon(s).</source>
        <translation>這將捨棄 {count} 個多邊形。</translation>
    </message>
    <message>
        <source>Use 'Export to layer' to keep them.</source>
        <translation>請使用「Export 至圖層」來保留它們。</translation>
    </message>
    <message>
        <source>This will end the current segmentation session.</source>
        <translation>這將結束目前的分割工作階段。</translation>
    </message>
    <message>
        <source>Do you want to continue?</source>
        <translation>是否要繼續？</translation>
    </message>
    <message>
        <source>Edit saved polygon</source>
        <translation>編輯已儲存的多邊形</translation>
    </message>
    <message>
        <source>Warning: you are about to edit an already saved polygon.</source>
        <translation>警告：您即將編輯一個已儲存的多邊形。</translation>
    </message>
    <message>
        <source>New to AI Segmentation?</source>
        <translation>第一次使用 AI Segmentation？</translation>
    </message>
    <message>
        <source>Watch our tutorial</source>
        <translation>觀看我們的教學影片</translation>
    </message>
    <message>
        <source>Contact us</source>
        <translation>聯絡我們</translation>
    </message>
    <message>
        <source>Bug, question, feature request?</source>
        <translation>錯誤回報、問題或功能建議？</translation>
    </message>
    <message>
        <source>We'd love to hear from you!</source>
        <translation>我們很樂意聽取您的意見！</translation>
    </message>
    <message>
        <source>Copy email address</source>
        <translation>複製電子郵件地址</translation>
    </message>
    <message>
        <source>or</source>
        <translation>或</translation>
    </message>
    <message>
        <source>Book a video call</source>
        <translation>預約視訊通話</translation>
    </message>
    <message>
        <source>Tutorial</source>
        <translation>教學</translation>
    </message>
    <message>
        <source>Settings</source>
        <translation>設定</translation>
    </message>
    <message>
        <source>Help</source>
        <translation>說明</translation>
    </message>
    <message>
        <source>Terms</source>
        <translation>條款</translation>
    </message>
    <message>
        <source>Privacy</source>
        <translation>隱私權</translation>
    </message>
    <message>
        <source>Something not working?</source>
        <translation>遇到問題了嗎？</translation>
    </message>
    <message>
        <source>Copy your logs and send them to us, we'll look into it :)</source>
        <translation>複製您的日誌並寄給我們，我們會盡快處理 :)</translation>
    </message>
    <message>
        <source>Segment elements on raster images using AI</source>
        <translation>使用 AI 分割點陣影像中的地物</translation>
    </message>
    <message>
        <source>Copy your logs with the button below and send them to our email.</source>
        <translation>使用下方按鈕複製您的日誌，並寄送至我們的電子郵件。</translation>
    </message>
    <message>
        <source>We'll fix your issue :)</source>
        <translation>我們會修復您的問題 :)</translation>
    </message>
    <message>
        <source>1. Click to copy logs</source>
        <translation>1. 點擊以複製日誌</translation>
    </message>
    <message>
        <source>2. Click to send to {}</source>
        <translation>2. 點擊以寄送至 {}</translation>
    </message>
    <message>
        <source>Open email client</source>
        <translation>開啟電子郵件用戶端</translation>
    </message>
    <message>
        <source>Copied!</source>
        <translation>已複製！</translation>
    </message>
    <message>
        <source>SSL Certificate Error</source>
        <translation>SSL 憑證錯誤</translation>
    </message>
    <message>
        <source>Installation Blocked</source>
        <translation>安裝被封鎖</translation>
    </message>
    <message>
        <source>Click is outside the &apos;{layer}&apos; raster. To segment another raster, stop the current segmentation first.</source>
        <translation>點擊位置在「{layer}」點陣圖範圍之外。若要分割其他點陣圖，請先停止目前的分割。</translation>
    </message>
    <message>
        <source>Big update dropped — v{version} is here!</source>
        <translation>重大更新來了－v{version} 正式推出！</translation>
    </message>
    <message>
        <source>Grab it now</source>
        <translation>立即更新</translation>
    </message>
    <message>
        <source>{ext} format is not directly supported. GDAL is not available.
Please convert your raster to GeoTIFF (.tif) before using AI Segmentation.</source>
        <translation>不直接支援 {ext} 格式，GDAL 無法使用。
請先將您的點陣圖轉換為 GeoTIFF（.tif）格式，再使用 AI Segmentation。</translation>
    </message>
    <message>
        <source>Cannot open {ext} file. The format may not be supported by your QGIS installation.
Please convert your raster to GeoTIFF (.tif) before using AI Segmentation.</source>
        <translation>無法開啟 {ext} 檔案，您的 QGIS 安裝可能不支援此格式。
請先將您的點陣圖轉換為 GeoTIFF（.tif）格式，再使用 AI Segmentation。</translation>
    </message>
    <message>
        <source>Failed to read {ext} file: {error}
Please convert your raster to GeoTIFF (.tif) manually.</source>
        <translation>讀取 {ext} 檔案失敗：{error}
請手動將您的點陣圖轉換為 GeoTIFF（.tif）格式。</translation>
    </message>
    <message>
        <source>PyTorch Error</source>
        <translation>PyTorch 錯誤</translation>
    </message>
    <message>
        <source>PyTorch cannot load on Windows</source>
        <translation>PyTorch 在 Windows 上無法載入</translation>
    </message>
    <message>
        <source>The plugin requires Visual C++ Redistributables to run PyTorch.

Please download and install:
https://aka.ms/vs/17/release/vc_redist.x64.exe

After installation, restart QGIS and try again.</source>
        <translation>此外掛程式需要安裝 Visual C++ Redistributables 才能執行 PyTorch。

請下載並安裝：
https://aka.ms/vs/17/release/vc_redist.x64.exe

安裝完成後，請重新啟動 QGIS 並再試一次。</translation>
    </message>
    <message>
        <source>Prediction Error</source>
        <translation>預測錯誤</translation>
    </message>
    <message>
        <source>Segmentation failed</source>
        <translation>分割失敗</translation>
    </message>
    <message>
        <source>Layer data provider is not available.</source>
        <translation>圖層資料提供者無法使用。</translation>
    </message>
    <message>
        <source>Failed to fetch tiles from the online layer. Check your network connection.</source>
        <translation>無法從線上圖層取得圖磚，請檢查您的網路連線。</translation>
    </message>
    <message>
        <source>Online layer returned blank tiles for this area. Try panning to an area with data coverage.</source>
        <translation>線上圖層在此區域傳回空白圖磚。請嘗試平移至有資料覆蓋的區域。</translation>
    </message>
    <message>
        <source>Crop Error</source>
        <translation>裁切錯誤</translation>
    </message>
    <message>
        <source>No raster file path available. Please restart segmentation.</source>
        <translation>沒有可用的點陣圖檔案路徑，請重新啟動分割。</translation>
    </message>
    <message>
        <source>Encoding Error</source>
        <translation>編碼錯誤</translation>
    </message>
    <message>
        <source>Report a Bug</source>
        <translation>回報錯誤</translation>
    </message>
    <message>
        <source>Disconnected parts detected in your polygon.</source>
        <translation>偵測到您的多邊形中有不連續的部分。</translation>
    </message>
    <message>
        <source>For best accuracy, segment one element at a time.</source>
        <translation>為求最佳準確度，請一次分割一個地物。</translation>
    </message>
    <message>
        <source>Layer: {}</source>
        <translation>圖層：{}</translation>
    </message>
    <message>
        <source>Polygon saved! Click on another element to segment, or export your polygons.</source>
        <translation>多邊形已儲存！點擊其他地物繼續分割，或 Export 您的多邊形。</translation>
    </message>
    <message>
        <source>Disconnected parts detected. For best accuracy, segment one element at a time.</source>
        <translation>偵測到不連續的部分。為求最佳準確度，請一次分割一個地物。</translation>
    </message>
    <message>
        <source>No element detected at this point. Try clicking on a different area.</source>
        <translation>此處未偵測到任何地物，請嘗試點擊其他區域。</translation>
    </message>
    <message>
        <source>Updating...</source>
        <translation>正在更新...</translation>
    </message>
    <message>
        <source>Check for Updates</source>
        <translation>檢查更新</translation>
    </message>
    <message>
        <source>More from TerraLab...</source>
        <translation>更多 TerraLab 產品...</translation>
    </message>
    <message>
        <source>Missing Visual C++ Redistributable. Install it, restart your computer, then click Retry.</source>
        <translation>缺少 Visual C++ Redistributable。請安裝後重新啟動電腦，再點擊「重試」。</translation>
    </message>
    <message>
        <source>Retry</source>
        <translation>重試</translation>
    </message>
    <message>
        <source>Cannot Write Export</source>
        <translation>無法寫入 Export</translation>
    </message>
    <message>
        <source>Cannot create export directory '{path}': {reason}</source>
        <translation>無法建立 Export 目錄「{path}」：{reason}</translation>
    </message>
    <message>
        <source>The export directory '{path}' is not writable. Choose a different location.</source>
        <translation>Export 目錄「{path}」不可寫入，請選擇其他位置。</translation>
    </message>
    <message>
        <source>Loading AI model...</source>
        <translation>正在載入 AI 模型...</translation>
    </message>
    <message>
        <source>SAM model ready</source>
        <translation>SAM 模型已就緒</translation>
    </message>
    <message>
        <source>Ready</source>
        <translation>就緒</translation>
    </message>
    <message>
        <source>Model load failed</source>
        <translation>模型載入失敗</translation>
    </message>
    <message>
        <source>Click landed outside the current element — segment one element at a time. Saving the current selection and starting a new one.</source>
        <translation>點擊位置超出目前的地物範圍－請一次分割一個地物。目前的選取範圍已儲存，並開始新的選取。</translation>
    </message>
    <message>
        <source>New here?</source>
        <translation>第一次使用嗎？</translation>
    </message>
    <message>
        <source>Watch the tutorial</source>
        <translation>觀看教學影片</translation>
    </message>
    <message>
        <source>Network Connection Problem</source>
        <translation>網路連線問題</translation>
    </message>
    <message>
        <source>Your connection appears unstable or blocked. Check: (1) your internet is working, (2) QGIS > Settings > Options > Network has a proxy configured if you are on a corporate network, (3) your firewall allows connections to pypi.org and files.pythonhosted.org.</source>
        <translation>您的連線似乎不穩定或已被封鎖。請檢查：(1) 網路是否正常運作，(2) 若您在公司網路環境中，QGIS > 偏好設定 > 選項 > 網路是否已設定代理伺服器，(3) 防火牆是否允許連線至 pypi.org 及 files.pythonhosted.org。</translation>
    </message>
    <message>
        <source>Sign in to TerraLab</source>
        <translation>登入 TerraLab</translation>
    </message>
    <message>
        <source>Two steps to start using AI Segmentation</source>
        <translation>兩步驟開始使用 AI Segmentation</translation>
    </message>
    <message>
        <source>1. Sign up or sign in on terra-lab.ai to get your key</source>
        <translation>1. 在 terra-lab.ai 註冊或登入以取得您的金鑰</translation>
    </message>
    <message>
        <source>2. Paste your key below to activate</source>
        <translation>2. 在下方貼上您的金鑰以啟用</translation>
    </message>
    <message>
        <source>1. Sign up / Sign in</source>
        <translation>1. 註冊／登入</translation>
    </message>
    <message>
        <source>Get Your Key</source>
        <translation>取得您的金鑰</translation>
    </message>
    <message>
        <source>2. Paste your activation key</source>
        <translation>2. 貼上您的啟用金鑰</translation>
    </message>
    <message>
        <source>Sign in to get your key</source>
        <translation>登入以取得您的金鑰</translation>
    </message>
    <message>
        <source>Create your free TerraLab account or sign in, then copy your activation key from the dashboard.</source>
        <translation>建立免費的 TerraLab 帳戶或登入，然後從控制台複製您的啟用金鑰。</translation>
    </message>
    <message>
        <source>Activate</source>
        <translation>啟用</translation>
    </message>
    <message>
        <source>Please enter your activation key.</source>
        <translation>請輸入您的啟用金鑰。</translation>
    </message>
    <message>
        <source>Checking...</source>
        <translation>正在檢查...</translation>
    </message>
    <message>
        <source>Activation key verified!</source>
        <translation>啟用金鑰已驗證！</translation>
    </message>
    <message>
        <source>Invalid activation key.</source>
        <translation>啟用金鑰無效。</translation>
    </message>
    <message>
        <source>Cannot reach server. Check your internet connection.</source>
        <translation>無法連上伺服器，請檢查您的網路連線。</translation>
    </message>
    <message>
        <source>Signed in!</source>
        <translation>已登入！</translation>
    </message>
    <message>
        <source>AI Edit</source>
        <translation>AI Edit</translation>
    </message>
    <message>
        <source>Generate imagery with AI on map zones (opens AI Edit plugin)</source>
        <translation>在地圖範圍上使用 AI 生成影像（開啟 AI Edit 外掛程式）</translation>
    </message>
    <message>
        <source>Right-click must be inside the current selection area.</source>
        <translation>右鍵點擊必須在目前的選取範圍內。</translation>
    </message>
    <message>
        <source>Account Settings</source>
        <translation>帳戶設定</translation>
    </message>
    <message>
        <source>Loading account info...</source>
        <translation>正在載入帳戶資訊...</translation>
    </message>
    <message>
        <source>Manage account on terra-lab.ai</source>
        <translation>在 terra-lab.ai 管理帳戶</translation>
    </message>
    <message>
        <source>Show</source>
        <translation>顯示</translation>
    </message>
    <message>
        <source>Hide</source>
        <translation>隱藏</translation>
    </message>
    <message>
        <source>Change activation key</source>
        <translation>變更啟用金鑰</translation>
    </message>
    <message>
        <source>Plan</source>
        <translation>方案</translation>
    </message>
    <message>
        <source>Free</source>
        <translation>免費</translation>
    </message>
    <message>
        <source>Canceled</source>
        <translation>已取消</translation>
    </message>
    <message>
        <source>Email</source>
        <translation>電子郵件</translation>
    </message>
    <message>
        <source>Key</source>
        <translation>金鑰</translation>
    </message>
    <message>
        <source>Model File Corrupted</source>
        <translation>模型檔案損毀</translation>
    </message>
    <message>
        <source>Re-downloading Model</source>
        <translation>正在重新下載模型</translation>
    </message>
    <message>
        <source>The AI model file was corrupted and is being re-downloaded. Please try your selection again once it finishes.</source>
        <translation>AI 模型檔案已損毀，正在重新下載。下載完成後請重新選取一次。</translation>
    </message>
    <message>
        <source>The AI model file is corrupted but could not be removed automatically. Please delete this folder and restart QGIS:</source>
        <translation>AI 模型檔案已損毀，但無法自動移除。請刪除此資料夾並重新啟動 QGIS：</translation>
    </message>
    <message>
        <source>Segment your map with AI</source>
        <translation>使用 AI 分割您的地圖</translation>
    </message>
    <message>
        <source>Sign in / Sign up to start</source>
        <translation>登入／註冊以開始使用</translation>
    </message>
    <message>
        <source>Sign in via your browser to start using AI Segmentation</source>
        <translation>透過瀏覽器登入以開始使用 AI Segmentation</translation>
    </message>
    <message>
        <source>Open again</source>
        <translation>再次開啟</translation>
    </message>
    <message>
        <source>Didn't open? Open the page again</source>
        <translation>沒有開啟嗎？再次開啟頁面</translation>
    </message>
    <message>
        <source>Couldn't open your browser. Use the manual key option below.</source>
        <translation>無法開啟您的瀏覽器，請使用下方的手動金鑰選項。</translation>
    </message>
    <message>
        <source>Sign-in timed out. Click Connect to try again.</source>
        <translation>登入逾時。請點擊「連線」再試一次。</translation>
    </message>
    <message>
        <source>Sign-in was cancelled in the browser. Click Connect to try again.</source>
        <translation>已在瀏覽器中取消登入。請點擊「連線」再試一次。</translation>
    </message>
    <message>
        <source>Unexpected response from the server. Please try again.</source>
        <translation>伺服器傳回未預期的回應，請再試一次。</translation>
    </message>
    <message>
        <source>This account has no active AI Segmentation plan. Reactivate it on terra-lab.ai, then click Connect again.</source>
        <translation>此帳戶沒有生效中的 AI Segmentation 方案。請在 terra-lab.ai 重新啟用，再點擊「連線」。</translation>
    </message>
    <message>
        <source>Connecting AI Segmentation</source>
        <translation>正在連線 AI Segmentation</translation>
    </message>
    <message>
        <source>Cancelling sign-in</source>
        <translation>正在取消登入</translation>
    </message>
    <message>
        <source>Help / Report a problem</source>
        <translation>說明／回報問題</translation>
    </message>
    <message>
        <source>Report a problem</source>
        <translation>回報問題</translation>
    </message>
    <message>
        <source>Connected</source>
        <translation>已連線</translation>
    </message>
    <message>
        <source>Sign out</source>
        <translation>登出</translation>
    </message>
    <message>
        <source>Sign out of AI Segmentation?</source>
        <translation>要登出 AI Segmentation 嗎？</translation>
    </message>
    <message>
        <source>You can sign back in anytime from QGIS.</source>
        <translation>您可以隨時從 QGIS 重新登入。</translation>
    </message>
    <message>
        <source>Active</source>
        <translation>生效中</translation>
    </message>
    <message>
        <source>Free Trial</source>
        <translation>免費試用</translation>
    </message>
    <message>
        <source>Make this map presentation-ready</source>
        <translation>讓這份地圖立即可用於簡報</translation>
    </message>
    <message>
        <source>AI Edit: turn your imagery into presentation and planning visuals</source>
        <translation>AI Edit：將您的影像轉換為簡報與規劃視覺圖</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Mode selection</source>
        <translation>模式選擇</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Choose between Manual (local) and Automatic (cloud) segmentation</source>
        <translation>在手動（本機）與自動（雲端）分割之間選擇</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Stop the active segmentation before switching modes.</source>
        <translation>請先停止目前進行中的分割，才能切換模式。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Cancel the active detection before switching modes.</source>
        <translation>請先取消目前進行中的偵測，才能切換模式。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Sign in to use Automatic mode</source>
        <translation>登入以使用自動模式</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Your free detections are used up</source>
        <translation>您的免費偵測次數已用完</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Subscribe to keep detecting without limits:</source>
        <translation>訂閱以繼續無限制偵測：</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Subscribe to Pro</source>
        <translation>訂閱 Pro</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detect every building, tree, or road automatically</source>
        <translation>自動偵測每一棟建築物、每一棵樹或每一條道路</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>No installation required, no GPU, no limits</source>
        <translation>無需安裝、無需 GPU、無使用限制</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Built for large-scale digitization projects</source>
        <translation>專為大型數化專案而設計</translation>
    </message>
    <message>
        <location filename="../src/ui/zone_selection_maptool.py" line="0"/>
        <source>Clear this zone</source>
        <translation>清除此範圍</translation>
    </message>
    <message>
        <location filename="../src/ui/zone_selection_maptool.py" line="0"/>
        <source>Cancel the running detection first</source>
        <translation>請先取消進行中的偵測</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>What do you want to detect?</source>
        <translation>您想偵測什麼？</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Where should the AI look?</source>
        <translation>AI 應該在哪裡尋找？</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Change</source>
        <translation>變更</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>What to detect...</source>
        <translation>要偵測的物件...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Hold the left mouse button and drag to draw a box on the map.</source>
        <translation>按住滑鼠左鍵並拖曳，在地圖上繪製方框。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} tile(s) = {n} credit(s)</source>
        <translation>{n} 個圖磚＝{n} 點數</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Object: {obj}</source>
        <translation>物件：{obj}</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detecting "{obj}"...</source>
        <translation>正在偵測「{obj}」...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Ground resolution per pixel. A smaller value lets the model detect smaller objects.</source>
        <translation>每像素的地面解析度。數值越小，模型可偵測到的物件越小。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detail</source>
        <translation>細節</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Higher detail splits the zone into more tiles. Each tile costs 1 credit and captures smaller objects.</source>
        <translation>細節越高，範圍會被切分成更多圖磚。每個圖磚耗費 1 點數，並可捕捉更小的物件。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Zone too large - reduce the selection area</source>
        <translation>範圍過大－請縮小選取範圍</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detect objects</source>
        <translation>偵測物件</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Cancel detection</source>
        <translation>取消偵測</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Tile {current}/{total}</source>
        <translation>圖磚 {current}/{total}</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Sending to the AI…</source>
        <translation>正在傳送至 AI…</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>You're next · starting now…</source>
        <translation>輪到您了·即將開始…</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Spot reserved · starting in ~{eta}</source>
        <translation>已保留名額·約 {eta} 後開始</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Spot reserved · starting soon…</source>
        <translation>已保留名額·即將開始…</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>High demand · your spot is held…</source>
        <translation>需求量大·您的名額已保留…</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Spot reserved · starting in a few seconds…</source>
        <translation>已保留名額·數秒後開始…</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{s} seconds</source>
        <translation>{s} 秒</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{m} min</source>
        <translation>{m} 分鐘</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} credits remaining (resets {date})</source>
        <translation>剩餘 {n} 點數（將於 {date} 重設）</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} credits remaining</source>
        <translation>剩餘 {n} 點數</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} free detection(s) remaining</source>
        <translation>剩餘 {n} 次免費偵測</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Drawing...</source>
        <translation>正在繪製...</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>{n} free detection(s) remaining (lifetime)</source>
        <translation>剩餘 {n} 次免費偵測（終身額度）</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Upgrade to Pro on terra-lab.ai</source>
        <translation>在 terra-lab.ai 升級至 Pro</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Pro</source>
        <translation>Pro</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>{remaining} / {total} credits</source>
        <translation>{remaining} / {total} 點數</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>resets {date}</source>
        <translation>於 {date} 重設</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Free uses</source>
        <translation>免費使用次數</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Credits</source>
        <translation>點數</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Auto detection (live)</source>
        <translation>自動偵測（即時）</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Preparing tiles...</source>
        <translation>正在準備圖磚...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Could not render the zone. Try a smaller area or another layer.</source>
        <translation>無法繪製此範圍，請嘗試縮小範圍或使用其他圖層。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Cancelling...</source>
        <translation>正在取消...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Finishing the previous run, please wait a moment...</source>
        <translation>正在完成上一次的執行，請稍候...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Found {n} object(s) but could not save the result file. Check folder permissions and the log.</source>
        <translation>找到 {n} 個物件，但無法儲存結果檔案。請檢查資料夾權限與日誌。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Could not save the result file. Check the log.</source>
        <translation>無法儲存結果檔案，請查看日誌。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>No detections found. Try a different prompt or zoom level.</source>
        <translation>找不到任何偵測結果。請嘗試不同的提示詞或縮放層級。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Resume detection</source>
        <translation>繼續偵測</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Detection failed. Check your connection and try again.</source>
        <translation>偵測失敗，請檢查您的連線並再試一次。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Not enough credits to continue. The finished tiles are kept.</source>
        <translation>點數不足，無法繼續。已完成的圖磚會予以保留。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Zone too large. Reduce the area to 50 tiles or fewer.</source>
        <translation>範圍過大，請將範圍縮小至 50 個圖磚以內。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Automatic detection is temporarily unavailable. Please try again later.</source>
        <translation>自動偵測暫時無法使用，請稍後再試。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Draw a zone first. Automatic detection on online layers needs a zone.</source>
        <translation>請先繪製範圍。線上圖層的自動偵測需要指定範圍。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>The zone is outside the selected raster layer. Pick the right layer or redraw the zone.</source>
        <translation>範圍超出所選點陣圖圖層之外。請選擇正確的圖層或重新繪製範圍。</translation>
    </message>
    <message>
        <source>Next</source>
        <translation>下一步</translation>
    </message>
    <message>
        <source>Export to layer</source>
        <translation>Export 至圖層</translation>
    </message>
    <message>
        <source>{n} object(s) detected - adjust below, then export</source>
        <translation>偵測到 {n} 個物件－在下方調整，然後 Export</translation>
    </message>
    <message>
        <source>Exported {n} polygon(s) to {name}</source>
        <translation>已 Export {n} 個多邊形至 {name}</translation>
    </message>
    <message>
        <source>Round corners</source>
        <translation>圓角</translation>
    </message>
    <message>
        <source>Fill holes</source>
        <translation>填補孔洞</translation>
    </message>
    <message>
        <source>Expand/Shrink:</source>
        <translation>擴張／收縮：</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Less</source>
        <translation>較少</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>More</source>
        <translation>較多</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>This area is large for this detail level. Raise detail or zoom in for sharper detections.</source>
        <translation>對此細節層級而言，此範圍偏大。請提高細節或放大以取得更清晰的偵測結果。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>This zone is too large for sharp detections, even at maximum detail. Draw a smaller zone for the best results.</source>
        <translation>即使在最高細節下，此範圍仍過大而無法取得清晰的偵測結果。請繪製較小的範圍以取得最佳效果。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Available when detection finishes</source>
        <translation>偵測完成後即可使用</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Finishing up... {done}/{total}</source>
        <translation>正在完成... {done}/{total}</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Min size:</source>
        <translation>最小尺寸：</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Max size:</source>
        <translation>最大尺寸：</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Off</source>
        <translation>關閉</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>No limit</source>
        <translation>無限制</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Hide detections smaller than this ground area. Use it to drop tiny noise blobs. 0 = keep all.</source>
        <translation>隱藏面積小於此地面面積的偵測結果，可用於剔除微小的雜訊斑點。0＝全部保留。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Hide detections larger than this ground area. 0 = no limit.</source>
        <translation>隱藏面積大於此地面面積的偵測結果。0＝無限制。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Finer detail finds smaller objects and costs more credits.</source>
        <translation>更精細的細節可偵測更小的物件，但會耗費更多點數。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>≈ {n} credits</source>
        <translation>≈ {n} 點數</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Finish</source>
        <translation>完成</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} object(s) detected - adjust below, then Finish</source>
        <translation>偵測到 {n} 個物件－在下方調整，再按「完成」</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Out of credits. Keep what was found below, then Finish.</source>
        <translation>點數已用盡。請保留下方找到的結果，再按「完成」。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Saved {n} polygon(s) to {name}</source>
        <translation>已將 {n} 個多邊形儲存至 {name}</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Cannot reach the server. Check your internet connection.</source>
        <translation>無法連上伺服器，請檢查您的網路連線。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Server refused the connection.</source>
        <translation>伺服器拒絕連線。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Request timed out. Check your connection or try again.</source>
        <translation>請求逾時，請檢查您的連線或再試一次。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>SSL certificate error. Your network may be blocking secure connections.</source>
        <translation>SSL 憑證錯誤，您的網路可能封鎖了安全連線。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Proxy connection failed. Check QGIS proxy settings (Settings &gt; Options &gt; Network).</source>
        <translation>代理伺服器連線失敗。請檢查 QGIS 代理伺服器設定（偏好設定 &gt; 選項 &gt; 網路）。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Authentication failed. Please sign in again.</source>
        <translation>驗證失敗，請重新登入。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Network error. Check your internet connection.</source>
        <translation>網路錯誤，請檢查您的網路連線。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Checking your AI Segmentation subscription</source>
        <translation>正在檢查您的 AI Segmentation 訂閱</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Loading AI Segmentation settings</source>
        <translation>正在載入 AI Segmentation 設定</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Refreshing credits</source>
        <translation>正在重新整理點數</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Warming up AI Segmentation</source>
        <translation>正在準備 AI Segmentation</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>What do you want to segment?</source>
        <translation>您想分割什麼？</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>e.g. building, solar panel (in English)</source>
        <translation>例如：building、solar panel（請使用英文）</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Popular</source>
        <translation>熱門</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Library</source>
        <translation>範例庫</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Browse objects with before / after examples.</source>
        <translation>瀏覽附有前後對比範例的物件。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>The prompt is sent to the AI in English. Describe the object in 1-2 words (e.g. building, solar panel).</source>
        <translation>提示詞會以英文傳送給 AI。請用 1-2 個字描述物件（例如：building、solar panel）。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use just 1-2 words for the object.</source>
        <translation>物件名稱請只用 1-2 個字。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Type the object itself, not a sentence or question.</source>
        <translation>請直接輸入物件名稱，而非句子或問句。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Too generic. Draw an example instead, or use a concrete word like building.</source>
        <translation>太籠統了。請改為繪製範例，或使用具體的字詞，例如 building。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Name a concrete object, not how it looks.</source>
        <translation>請命名具體的物件，而非描述其外觀。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Segment one object - drop words like 'near' or 'with'.</source>
        <translation>請分割單一物件－請去除像「near」或「with」之類的字詞。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use a real object word.</source>
        <translation>請使用真實存在的物件名稱。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use a 1-2 word object name.</source>
        <translation>請使用 1-2 個字的物件名稱。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Did you mean '{term}'?</source>
        <translation>您是指「{term}」嗎？</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Loading...</source>
        <translation>正在載入...</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>No preview</source>
        <translation>無預覽</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>No preview yet</source>
        <translation>尚無預覽</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Segment library</source>
        <translation>分割範例庫</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>RECENT</source>
        <translation>最近</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Recently detected</source>
        <translation>最近偵測</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>BROWSE</source>
        <translation>瀏覽</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Objects you detect will appear here.</source>
        <translation>您偵測過的物件會顯示於此。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>today</source>
        <translation>今天</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>yesterday</source>
        <translation>昨天</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>{n} days ago</source>
        <translation>{n} 天前</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>{n} detection(s)</source>
        <translation>{n} 次偵測</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>{n} object(s)</source>
        <translation>{n} 個物件</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Loading segment library</source>
        <translation>正在載入分割範例庫</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Pick an object to see a before / after, then use it.</source>
        <translation>選擇一個物件以檢視前後對比，然後套用它。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Search objects... e.g. building, solar panel</source>
        <translation>搜尋物件... 例如：building、solar panel</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Prompt:</source>
        <translation>提示詞：</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Fuzzy edges: this one may need cleanup after detection.</source>
        <translation>邊緣模糊：此物件偵測後可能需要清理。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use this prompt</source>
        <translation>使用此提示詞</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>No matching objects.</source>
        <translation>沒有符合的物件。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use</source>
        <translation>使用</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>≈ {n} tiles = {n} credits</source>
        <translation>≈ {n} 個圖磚＝{n} 點數</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Automatic mode scans your zone tile by tile. 1 tile = 1 credit, so this run costs about {n} credits. More detail splits the zone into more tiles, which costs more credits.</source>
        <translation>自動模式會逐一圖磚掃描您的範圍。1 個圖磚＝1 點數，因此此次執行約需 {n} 點數。細節越高，範圍會切分成越多圖磚，耗費的點數也越多。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Draw your example inside the selected zone.</source>
        <translation>在所選範圍內繪製您的範例。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Clean edges:</source>
        <translation>清理邊緣：</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Remove thin ragged fringe attached to the outline (0 = no change)</source>
        <translation>移除輪廓上的細碎鋸齒邊緣（0＝不變更）</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Share anonymous usage statistics</source>
        <translation>分享匿名使用統計資料</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Helps us fix bugs faster. Never includes your data, layers or coordinates.</source>
        <translation>協助我們更快修復錯誤，絕不包含您的資料、圖層或座標。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} objects found</source>
        <translation>找到 {n} 個物件</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>No objects found</source>
        <translation>未找到任何物件</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>All shown at {pct}% confidence</source>
        <translation>全部以 {pct}% 信心度顯示</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{visible} shown at {pct}% · {hidden} more below this confidence</source>
        <translation>已顯示 {visible} 個（{pct}%）·另有 {hidden} 個低於此信心度</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>0 shown at {pct}% - lower Confidence to reveal them</source>
        <translation>在 {pct}% 下顯示 0 個－請降低信心度以顯示它們</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Started at {pct}% - nothing scored above.</source>
        <translation>從 {pct}% 開始－沒有更高分的結果。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>More objects</source>
        <translation>更多物件</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Only confident</source>
        <translation>僅顯示確定的</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Objects not quite right? Refine them in Manual mode.</source>
        <translation>物件不太準確？請在手動模式中細修。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Export {n} polygons</source>
        <translation>Export {n} 個多邊形</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Lower Confidence to show objects first.</source>
        <translation>請先降低信心度以顯示物件。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Adjust &amp; run again</source>
        <translation>調整並重新執行</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Discard these detections?</source>
        <translation>要捨棄這些偵測結果嗎？</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Your {total} detections will be discarded. You keep your zone, object and settings. Running Detect again will use new credits.</source>
        <translation>您的 {total} 個偵測結果將被捨棄。範圍、物件與設定會保留。重新執行偵測會耗費新的點數。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Discard &amp;&amp; adjust</source>
        <translation>捨棄並調整</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Keep your detections?</source>
        <translation>要保留您的偵測結果嗎？</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Save {visible} detections to a layer before leaving?</source>
        <translation>離開前要將 {visible} 個偵測結果儲存至圖層嗎？</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Save {total} detections (currently hidden by Confidence) to a layer before leaving?</source>
        <translation>離開前要將 {total} 個偵測結果（目前因信心度設定而隱藏）儲存至圖層嗎？</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Save &amp;&amp; exit</source>
        <translation>儲存並離開</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Discard &amp;&amp; exit</source>
        <translation>捨棄並離開</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>No {object} found in this zone</source>
        <translation>在此範圍內未找到任何 {object}</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>This run used {n} credits. Things that usually fix it:</source>
        <translation>此次執行耗費了 {n} 點數。以下方法通常有幫助：</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Check the word is English and singular (building, not batiments)</source>
        <translation>確認字詞為英文且為單數形式（例如 building，而非 batiments）</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Draw an example of one object (step 2)</source>
        <translation>繪製一個物件的範例（步驟 2）</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Raise Detail so small objects are visible</source>
        <translation>提高細節以顯示較小的物件</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Try a smaller or different zone</source>
        <translation>嘗試使用較小或不同的範圍</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detecting "{obj}"... · {n} found so far</source>
        <translation>正在偵測「{obj}」...·目前已找到 {n} 個</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>How many objects sit at each confidence level.</source>
        <translation>各信心度層級所包含的物件數量。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Your 500 free detections are used up</source>
        <translation>您的 500 次免費偵測已用完</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>10,000 detections every month (~1,700 km2)</source>
        <translation>每月 10,000 次偵測（約 1,700 平方公里）</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Every building, tree, or road as clean polygons</source>
        <translation>每一棟建築物、每一棵樹或每一條道路都會以乾淨的多邊形呈現</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Cancel anytime; your exported layers stay yours</source>
        <translation>隨時可取消；已 Export 的圖層永遠屬於您</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Opens your TerraLab dashboard</source>
        <translation>開啟您的 TerraLab 控制台</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Start Automatic AI Segmentation</source>
        <translation>開始自動 AI Segmentation</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Draw your zone</source>
        <translation>繪製您的範圍</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Click on the map to outline the area to scan.</source>
        <translation>點擊地圖以圈選要掃描的區域。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Keep clicking around the area, at least 3 points.</source>
        <translation>持續點擊以圈選區域，至少需要 3 個點。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Click the first point to close the zone.</source>
        <translation>點擊起點以封閉範圍。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>undo point</source>
        <translation>復原點</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Now outline one object, then click the first point to close.</source>
        <translation>現在圈選一個物件，再點擊起點以封閉範圍。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Click on the map to drop points around the area you want to scan.</source>
        <translation>點擊地圖以在要掃描的區域周圍放置點。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Exit</source>
        <translation>退出</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>e.g. building, tree, road, car</source>
        <translation>例如：building、tree、road、car</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Browse ready-to-use objects with before / after previews.</source>
        <translation>瀏覽附有前後對比預覽的現成物件。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Draw an example</source>
        <translation>繪製範例</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Subscribe to finish this zone: 10,000 credits/month.</source>
        <translation>訂閱以完成此範圍：每月 10,000 點數。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Blue = detected object</source>
        <translation>藍色＝偵測到的物件</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Filter detections by confidence. Lower shows more (and noisier), higher keeps only the strongest. Free and instant.</source>
        <translation>依信心度篩選偵測結果。數值越低顯示越多（但雜訊也越多），數值越高則只保留最可靠的結果。免費且即時。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Show tiles (debug)</source>
        <translation>顯示圖磚（偵錯）</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Open these detections in Manual mode to fix specific objects point-by-point, then come back and export.</source>
        <translation>在手動模式中開啟這些偵測結果，逐點修正特定物件，完成後返回並進行 Export。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Try instead:</source>
        <translation>請改試：</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>"{word}" will run as "{token}".</source>
        <translation>「{word}」將以「{token}」執行。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>That word isn't recognized - try a common object like building or tree.</source>
        <translation>無法辨識該字詞－請嘗試常見的物件，例如 building 或 tree。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>One object per run - start with the first one, then run again.</source>
        <translation>每次執行僅限一種物件－請先處理第一種，再重新執行。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>The Library has ready-to-use objects.</source>
        <translation>範例庫中有現成的物件可供使用。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Keep clicking to add points ({n} so far, 3 minimum).</source>
        <translation>持續點擊以新增點（目前 {n} 個，最少需 3 個）。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>{n} points. Double-click or press Enter to finish, or click the first point to close.</source>
        <translation>已有 {n} 個點。雙擊或按 Enter 完成，或點擊起點以封閉範圍。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>~ {credits} credits</source>
        <translation>~ {credits} 點數</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>{remaining} left</source>
        <translation>剩餘 {remaining}</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>{remaining} free left</source>
        <translation>剩餘 {remaining} 個免費</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>~ {credits} credits · </source>
        <translation>~ {credits} 點數 · </translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>will stop after {remaining}</source>
        <translation>將於 {remaining} 後停止</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>1 credit ~ 0.17 km2 at default detail.</source>
        <translation>在預設細節下，1 點數約等於 0.17 平方公里。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Tip: draw an example of one {object} to catch more of them.</source>
        <translation>提示：繪製一個 {object} 的範例，可找到更多類似物件。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>object</source>
        <translation>物件</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Dense area: raise Detail to catch every object.</source>
        <translation>密集區域：請提高細節以捕捉所有物件。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/cards.py" line="0"/>
        <source>Restore</source>
        <translation>還原</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/cards.py" line="0"/>
        <source>Delete forever</source>
        <translation>永久刪除</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/cards.py" line="0"/>
        <source>Deleted {when} · purges in {n} days</source>
        <translation>已於 {when} 刪除·將於 {n} 天後清除</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/cards.py" line="0"/>
        <source>{tiles} tiles · {objects} objects · {credits} credits</source>
        <translation>{tiles} 個圖磚·{objects} 個物件·{credits} 點數</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/cards.py" line="0"/>
        <source>Older detection</source>
        <translation>較早的偵測結果</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Details</source>
        <translation>詳細資訊</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Fullscreen</source>
        <translation>全螢幕</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Exit fullscreen</source>
        <translation>退出全螢幕</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Prompt</source>
        <translation>提示詞</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Copy</source>
        <translation>複製</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Copy prompt</source>
        <translation>複製提示詞</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Copied</source>
        <translation>已複製</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Template</source>
        <translation>範本</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Your detection</source>
        <translation>您的偵測結果</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Open the Library from the Automatic page to use this.</source>
        <translation>請從自動頁面開啟範例庫以使用此功能。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>DATE</source>
        <translation>日期</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>OBJECTS</source>
        <translation>物件</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>CREDITS</source>
        <translation>點數</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>TILES</source>
        <translation>圖磚</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>RESOLUTION</source>
        <translation>解析度</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>EXAMPLE</source>
        <translation>範例</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Used</source>
        <translation>已使用</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Restore to map</source>
        <translation>還原至地圖</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Reopens this run's review at the same place. Free - no credits.</source>
        <translation>在原處重新開啟此次執行的檢視畫面。免費－不耗費點數。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Export...</source>
        <translation>Export...</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Delete</source>
        <translation>刪除</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Drop this object back into the prompt box for a new detection.</source>
        <translation>將此物件放回提示詞欄位，以進行新的偵測。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Remove from favorites</source>
        <translation>從收藏中移除</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Add to favorites</source>
        <translation>加入收藏</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Format:</source>
        <translation>格式：</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>GeoPackage keeps the embedded style; other formats are saved without a style.</source>
        <translation>GeoPackage 會保留內嵌的樣式；其他格式則不含樣式儲存。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Browse...</source>
        <translation>瀏覽...</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Your detections</source>
        <translation>您的偵測結果</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Recent</source>
        <translation>最近</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Favorites</source>
        <translation>收藏</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Recently deleted</source>
        <translation>最近刪除</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Templates</source>
        <translation>範本</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Load older runs</source>
        <translation>載入較早的執行紀錄</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Nothing here yet. Your automatic detections will land here, ready to reuse, restore or export.</source>
        <translation>目前尚無內容。您的自動偵測結果會顯示於此，可重複使用、還原或 Export。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Star a detection to keep it here.</source>
        <translation>將偵測結果加入收藏即可保留在此。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Deleted runs wait here for 30 days, then they are purged for good.</source>
        <translation>已刪除的執行紀錄會在此保留 30 天，之後將永久清除。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>This permanently removes the stored previews and masks. Exported layers are never touched.</source>
        <translation>這會永久移除已儲存的預覽與遮罩。已 Export 的圖層絕不會受影響。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Could not load this run's stored detections. Try again later.</source>
        <translation>無法載入此次執行儲存的偵測結果，請稍後再試。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Nothing to export at this confidence. Lower it and try again.</source>
        <translation>在此信心度下沒有可 Export 的內容，請降低後再試一次。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>The export failed. Check the file path and try again.</source>
        <translation>Export 失敗，請檢查檔案路徑並再試一次。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Exported {n} polygon(s).</source>
        <translation>已 Export {n} 個多邊形。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Add a point</source>
        <translation>新增一個點</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Add area</source>
        <translation>新增區域</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>All detections kept. Go 'Back to review' to export.</source>
        <translation>所有偵測結果均已保留。請點擊「返回檢視」進行 Export。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Arrow keys</source>
        <translation>方向鍵</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Automatic</source>
        <translation>自動</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Automatic - detect and review</source>
        <translation>自動－偵測與檢視</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Automatic - draw your zone</source>
        <translation>自動－繪製您的範圍</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Cancel the drawing</source>
        <translation>取消繪製</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Cancel the running detection, or exit the review</source>
        <translation>取消進行中的偵測，或退出檢視</translation>
    </message>
    <message>
        <location filename="../src/core/feature_encoder.py" line="0"/>
        <source>Click</source>
        <translation>點擊</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_run.py" line="0"/>
        <source>Could not place the example on the image. Redraw the example box inside the zone and try again.</source>
        <translation>無法將範例放置於影像上。請在範圍內重新繪製範例方框並再試一次。</translation>
    </message>
    <message>
        <location filename="../src/core/run_restore.py" line="0"/>
        <source>Could not rebuild this run's detections.</source>
        <translation>無法重建此次執行的偵測結果。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Delete removes this object</source>
        <translation>按 Delete 可移除此物件</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Delete the active object</source>
        <translation>刪除目前的物件</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Detect objects, or export the reviewed polygons</source>
        <translation>偵測物件，或 Export 已檢視的多邊形</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detection continues in the background. Reopen AI Segmentation to follow it.</source>
        <translation>偵測將在背景中繼續進行。重新開啟 AI Segmentation 即可追蹤進度。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Double-click</source>
        <translation>雙擊</translation>
    </message>
    <message>
        <location filename="../src/core/pip_diagnostics.py" line="0"/>
        <source>Example</source>
        <translation>範例</translation>
    </message>
    <message>
        <location filename="../src/core/run_restore.py" line="0"/>
        <source>Finish or exit the current run before restoring a past one.</source>
        <translation>請先完成或退出目前的執行，才能還原過去的執行紀錄。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Finish the zone</source>
        <translation>完成範圍</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>General</source>
        <translation>一般</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Hand-refined objects are always kept, whatever the confidence.</source>
        <translation>手動細修過的物件永遠會保留，無論信心度為何。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/manual_handoff.py" line="0"/>
        <source>Install now</source>
        <translation>立即安裝</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Keeps this polygon in your session. Export writes all kept polygons to a layer.</source>
        <translation>將此多邊形保留在此工作階段中。Export 會將所有已保留的多邊形寫入圖層。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_maptool.py" line="0"/>
        <source>Keyboard shortcuts</source>
        <translation>鍵盤快速鍵</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Left-click</source>
        <translation>左鍵點擊</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Left-click adds area · Right-click removes area</source>
        <translation>左鍵點擊新增區域·右鍵點擊移除區域</translation>
    </message>
    <message>
        <location filename="../src/core/layer_conventions.py" line="0"/>
        <source>Manual</source>
        <translation>手動</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/manual_handoff.py" line="0"/>
        <source>Manual mode needs a one-time setup</source>
        <translation>手動模式需要進行一次性設定</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Manual session</source>
        <translation>手動工作階段</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Max detail for this zone - draw a larger zone for finer detail.</source>
        <translation>此範圍已達最高細節－請繪製較大的範圍以取得更精細的細節。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Merges your edits back into the review. Nothing is exported yet.</source>
        <translation>將您的編輯合併回檢視畫面，尚未進行任何 Export。</translation>
    </message>
    <message>
        <location filename="../src/core/venv_manager.py" line="0"/>
        <source>OK</source>
        <translation>確定</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>One color per object - check neighbors are separated</source>
        <translation>每個物件使用不同顏色－請確認相鄰物件已明確區分</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Optional shape and size controls: simplify outlines, clean edges, round corners, expand or shrink, fill holes, size filters.</source>
        <translation>選用的形狀與大小控制項：簡化輪廓、清理邊緣、圓角、擴張或收縮、填補孔洞、大小篩選。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_lifecycle.py" line="0"/>
        <source>Out of credits after {done}/{total} tiles. Your detections are kept below.</source>
        <translation>在完成 {done}/{total} 個圖磚後點數已用盡。您的偵測結果會保留於下方。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_results.py" line="0"/>
        <source>Outlines only - check boundaries against the imagery</source>
        <translation>僅顯示輪廓－請對照影像確認邊界</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Pan the map</source>
        <translation>平移地圖</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Part of your zone is outside "{layer}" - only the overlapping area will return objects.</source>
        <translation>您的部分範圍位於「{layer}」之外－僅重疊區域會傳回偵測結果。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_run.py" line="0"/>
        <source>Pick an object to detect first (nothing was selected).</source>
        <translation>請先選擇要偵測的物件（目前尚未選取任何物件）。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Polygon saved ({n} total). Click another element, or export when done.</source>
        <translation>多邊形已儲存（共 {n} 個）。點擊其他地物，或完成後 Export。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Press S to keep it (turns green)</source>
        <translation>按 S 保留它（會變成綠色）</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Refine detections</source>
        <translation>細修偵測結果</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_results.py" line="0"/>
        <source>Refine seeds</source>
        <translation>細修種子</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/manual_handoff.py" line="0"/>
        <source>Refining uses the free local AI, which is not installed yet. Install it now (a few minutes, automatic)? Your detections stay safely in the review.</source>
        <translation>細修功能會使用免費的本機 AI，但尚未安裝。要立即安裝嗎？（僅需幾分鐘，全自動）您的偵測結果會安全保留在檢視畫面中。</translation>
    </message>
    <message>
        <location filename="../src/core/checkpoint_manager.py" line="0"/>
        <source>Remove</source>
        <translation>移除</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Remove area</source>
        <translation>移除區域</translation>
    </message>
    <message>
        <location filename="../src/ui/zone_selection_maptool.py" line="0"/>
        <source>Remove zone</source>
        <translation>移除範圍</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Resets {date}</source>
        <translation>於 {date} 重設</translation>
    </message>
    <message>
        <location filename="../src/core/run_restore.py" line="0"/>
        <source>Restored "{prompt}" - adjust and export below.</source>
        <translation>已還原「{prompt}」－請在下方調整並 Export。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Right-click</source>
        <translation>右鍵點擊</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Save polygon (S)</source>
        <translation>儲存多邊形（S）</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_lifecycle.py" line="0"/>
        <source>Session expired. Sign in again to continue.</source>
        <translation>工作階段已逾期，請重新登入以繼續。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Start (the visible mode's Start button)</source>
        <translation>開始（目前顯示模式的「開始」按鈕）</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Start Manual AI Segmentation</source>
        <translation>開始手動 AI Segmentation</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_lifecycle.py" line="0"/>
        <source>The raster was removed. Your polygons were saved to a layer.</source>
        <translation>點陣圖已被移除，您的多邊形已儲存至圖層。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_lifecycle.py" line="0"/>
        <source>The selected raster was removed.</source>
        <translation>所選的點陣圖已被移除。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>The selected raster was removed. Keeping what was already found.</source>
        <translation>所選的點陣圖已被移除，已找到的結果會予以保留。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Tip: S saves, Enter exports, Ctrl+Z undoes a click.</source>
        <translation>提示：S 鍵儲存，Enter 鍵 Export，Ctrl+Z 復原上一次點擊。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Undo the last point</source>
        <translation>復原上一個點</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Writes a GeoPackage layer with your {n} kept polygons.</source>
        <translation>將您保留的 {n} 個多邊形寫入 GeoPackage 圖層。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_results.py" line="0"/>
        <source>Yellow = confident · Purple = uncertain</source>
        <translation>黃色＝確定·紫色＝不確定</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Your zone is outside "{layer}". Pick the right layer or draw inside it.</source>
        <translation>您的範圍位於「{layer}」之外，請選擇正確的圖層或在其範圍內繪製。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_zone.py" line="0"/>
        <source>Zone too large. Reduce the area to {max} tiles or fewer.</source>
        <translation>範圍過大，請將範圍縮小至 {max} 個圖磚以內。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>objects</source>
        <translation>物件</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{kept} of {total} detections kept - click a blue detection to edit it</source>
        <translation>已保留 {kept}／{total} 個偵測結果－點擊藍色的偵測結果以編輯</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} of {total} free detections left</source>
        <translation>剩餘 {n}／{total} 次免費偵測</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{remaining} free detections left.</source>
        <translation>剩餘 {remaining} 次免費偵測。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>≈ 1 tile = 1 credit</source>
        <translation>≈ 1 個圖磚＝1 點數</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Keep this result</source>
        <translation>保留此結果</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Start over</source>
        <translation>重新開始</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Adjust and run again</source>
        <translation>調整並重新執行</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>all shown</source>
        <translation>全部顯示</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>{visible} of {n} shown</source>
        <translation>已顯示 {visible}／{n}</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>{hidden} below {pct}%</source>
        <translation>{hidden} 個低於 {pct}%</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Adjust and run again</source>
        <translation>調整並重新執行</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Keep this result</source>
        <translation>保留此結果</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Start over</source>
        <translation>重新開始</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>all shown</source>
        <translation>全部顯示</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>{hidden} below {pct}%</source>
        <translation>{hidden} 個低於 {pct}%</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>{visible} of {n} shown</source>
        <translation>已顯示 {visible}／{n}</translation>
    </message>
    <message>
        <source>Right angles:</source>
        <translation>直角化：</translation>
    </message>
    <message>
        <source>Snap edges to 90 degrees for man-made shapes like buildings, pools and solar panels.</source>
        <translation>將邊緣校正為 90 度角，適用於建築物、泳池、太陽能板等人造形狀。</translation>
    </message>
    <message>
        <source>Click an object on the map and the AI outlines it. You go one object at a time, checking and saving each polygon yourself.</source>
        <translation>點擊地圖上的物件，AI 會自動描繪其輪廓。您將逐一處理每個物件，親自檢查並儲存每個多邊形。</translation>
    </message>
    <message>
        <source>Show guidance tips again</source>
        <translation>再次顯示操作提示</translation>
    </message>
    <message>
        <source>Guidance tips restored</source>
        <translation>操作提示已還原</translation>
    </message>
    <message>
        <source>Run again here</source>
        <translation>在此重新執行</translation>
    </message>
    <message>
        <source>Reload this zone and object, ready to detect.</source>
        <translation>重新載入此範圍與物件，準備進行偵測。</translation>
    </message>
    <message>
        <source>Same object, new zone</source>
        <translation>相同物件，新範圍</translation>
    </message>
    <message>
        <source>Keep this object and draw a new zone on the map.</source>
        <translation>保留此物件，並在地圖上繪製新的範圍。</translation>
    </message>
    <message>
        <source>Upgrade to Pro</source>
        <translation>升級至 Pro</translation>
    </message>
    <message>
        <source>Free account - sign up takes 15 seconds in your browser.</source>
        <translation>免費帳戶－在瀏覽器中註冊僅需 15 秒。</translation>
    </message>
    <message>
        <source>Manual mode stays free and unlimited on your computer.</source>
        <translation>手動模式在您的電腦上永久免費且無使用限制。</translation>
    </message>
    <message>
        <source>Finds every object of one kind in your zone - draw a zone, name the object, get all the polygons at once.</source>
        <translation>找出您範圍內同一類型的所有物件－繪製範圍、命名物件，即可一次取得所有多邊形。</translation>
    </message>
    <message>
        <source>Tip: draw an example on the map to boost detection of unusual objects.</source>
        <translation>提示：在地圖上繪製範例，可提升對不常見物件的偵測效果。</translation>
    </message>
    <message>
        <source>Tip: lower Confidence to reveal more detections, raise it to keep only the best.</source>
        <translation>提示：降低信心度可顯示更多偵測結果，提高信心度則只保留最佳結果。</translation>
    </message>
    <message>
        <source>This zone is {area} km2 - free trial zones go up to {max} km2.</source>
        <translation>此範圍為 {area} 平方公里－免費試用範圍上限為 {max} 平方公里。</translation>
    </message>
    <message>
        <source>Draw a smaller zone, or &lt;a href=&quot;{url}&quot;&gt;subscribe&lt;/a&gt; to segment areas of any size.</source>
        <translation>請繪製較小的範圍，或&lt;a href=&quot;{url}&quot;&gt;訂閱&lt;/a&gt;以分割任意大小的區域。</translation>
    </message>
    <message>
        <source>Running low: {n} free detections left. &lt;a href=&quot;{url}&quot;&gt;Subscribe&lt;/a&gt; to keep going.</source>
        <translation>額度即將用盡：剩餘 {n} 次免費偵測。&lt;a href=&quot;{url}&quot;&gt;訂閱&lt;/a&gt;以繼續使用。</translation>
    </message>
    <message>
        <source>Last run: {count} {object} exported · {area} km2 · {used} credits used, {left} left</source>
        <translation>上次執行：已 Export {count} 個 {object}·{area} 平方公里·已使用 {used} 點數，剩餘 {left}</translation>
    </message>
    <message>
        <source>Last run: {count} {object} exported · {area} km2 · {used} credits used</source>
        <translation>上次執行：已 Export {count} 個 {object}·{area} 平方公里·已使用 {used} 點數</translation>
    </message>
    <message>
        <source>Then segment any imagery: point and click, or fully automatic.</source>
        <translation>接著即可分割任何影像：點擊選取，或完全自動化。</translation>
    </message>
    <message>
        <source>Waiting for your browser sign-in...</source>
        <translation>正在等待您於瀏覽器中登入...</translation>
    </message>
    <message>
        <source>New: Automatic mode finds every object in a zone at once.</source>
        <translation>新功能：自動模式可一次找出範圍內的所有物件。</translation>
    </message>
    <message>
        <source>Try Automatic</source>
        <translation>試用自動模式</translation>
    </message>
    <message>
        <source>Got it - hide this tip</source>
        <translation>了解了－隱藏此提示</translation>
    </message>
    <message>
        <source>Finish or cancel the current detection before re-running a past one.</source>
        <translation>請先完成或取消目前的偵測，才能重新執行過去的紀錄。</translation>
    </message>
    <message>
        <source>Couldn&apos;t open your browser. Check your connection and click Sign in / Sign up to start again.</source>
        <translation>無法開啟您的瀏覽器。請檢查您的連線，再點擊「登入／註冊」重新開始。</translation>
    </message>
</context>
</TS>
