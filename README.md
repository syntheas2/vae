# Schrittweise Erklärung des Diffusionsmodells in TabSyn

## 1. Vorbereitungsschritt: VAE für latente Repräsentation
- Der VAE (Variational Autoencoder) komprimiert die Tabellendaten in einen latenten Raum
- Aus den Eingabedaten `x` wird eine latente Variable `z` gelernt
- Der VAE besteht aus einem Encoder (x → z) und einem Decoder (z → x̂)
- Die latente Variable `z` ist eine kompakte, kontinuierliche Repräsentation der ursprünglichen Daten
- Im Code: `Model_VAE` Klasse mit `Encoder` und `Decoder`

## 2. Grundprinzip des Diffusionsmodells
- Das Diffusionsmodell arbeitet im latenten Raum mit den `z`-Werten
- Zwei Kernprozesse: Forward Process (Hinzufügen von Rauschen) und Reverse Process (Entfernen von Rauschen)
- Ziel: Ein Modell trainieren, das den Reverse Process beherrscht, um später Daten zu generieren

## 3. Forward Process (Verrauschung)
- Systematisches Hinzufügen von Rauschen zu den latenten Variablen `z`
- Im Code: `n = torch.randn_like(y) * sigma` - Gauß'sches Rauschen wird skaliert mit `sigma`
- `sigma` wird aus einer Zufallsverteilung gemäß `P_mean` und `P_std` gezogen
- Diese verrauschten Daten `y + n` dienen als Eingabe für das Trainingsmodell

## 4. Training des Reverse Process
- Das Diffusionsmodell lernt die Funktion: (verrauschte Daten, Rauschstärke) → originale Daten
- Im Code implementiert durch: `D_yn = denoise_fn(y + n, sigma)`
- Die Verlustfunktion `EDMLoss` misst den Unterschied zwischen Vorhersage und Original
- Während des Trainings werden verschiedene Rauschstärken verwendet, um Robustheit zu erreichen
- Optimierungsschritt: `optimizer.step()` aktualisiert die Parameter des Modells basierend auf dem Verlust

## 5. Speichern des trainierten Modells
- Das Modell mit dem besten Trainingsverlust wird gespeichert: `torch.save(model.state_dict(), model_save_path)`
- Regelmäßige Checkpoints werden erstellt: `torch.save(model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')`

## 6. Datensynthese mit dem trainierten Modell
- Für die Synthese wird der gelernte Reverse Process verwendet
- Startpunkt: Zufällig generiertes Rauschen im latenten Raum
- Schrittweise Anwendung des Diffusionsmodells zum Entfernen des Rauschens
- Implementiert in der `sample`-Funktion: Von hohem Rauschen (`sigma_max`) zu niedrigem Rauschen (`sigma_min`)
- Das entstörte Ergebnis wird durch den VAE-Decoder zurück in den Datenraum transformiert

## 7. Warum Training mit Trainingsdaten evaluiert wird
- Das Diffusionsmodell lernt die Rauschentfernung aus bekannten Daten
- Die Qualität der Rauschentfernung wird direkt auf den Trainingsdaten gemessen
- Der Trainingsverlust `curr_loss` reflektiert die Fähigkeit zur präzisen Rauschentfernung
- Validierung erfolgt implizit durch die Bewertung der generierten Samples nach dem Training

## 8. Besonderheiten in der TabSyn-Implementierung
- Verwendet die EDM-Verlustfunktion für stabileres Training
- Komplexe Gewichtung des Verlusts basierend auf dem Rausch-Signal-Verhältnis
- Implementation des MLPDiffusion-Modells mit speziellen Positional Embeddings für die Rauschstärke
- Frühe Stoppbedingung, wenn sich der Trainingsverlust nicht mehr verbessert