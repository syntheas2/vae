# Erklärung der VAE-Metriken

- **beta**
  - Gewichtungsfaktor für KL-Divergenz im Beta-VAE
  - ⬆️ OFT STEIGEND - wird häufig während des Trainings erhöht (Annealing)

- **Train MSE**
  - Mean Squared Error für numerische Daten
  - ⬇️ FALLEND - niedrigere Werte bedeuten bessere Rekonstruktion

- **Train CE**
  - Cross-Entropy Verlust für kategorische Daten
  - ⬇️ FALLEND - niedrigere Werte bedeuten bessere Rekonstruktion

- **Train KL**
  - Kullback-Leibler Divergenz zwischen latenter und Zielverteilung
  - ⚖️ BALANCIERT - sollte sich mit der Zeit stabilisieren
  - Kann mit steigendem beta zunehmen

- **Val MSE**
  - Validierungs-MSE für numerische Daten
  - ⬇️ FALLEND - sollte parallel zu Train MSE sinken
  - Sollte nicht deutlich höher als Train MSE sein

- **Val CE**
  - Validierungs-Cross-Entropy für kategorische Daten
  - ⬇️ FALLEND - sollte parallel zu Train CE sinken
  - Sollte nicht deutlich höher als Train CE sein

- **Train ACC**
  - Trainingsgenauigkeit für kategorische Vorhersagen
  - ⬆️ STEIGEND - höhere Werte bedeuten bessere Klassifikation
  - Berechnung: Anteil der korrekt vorhergesagten Kategorien (Anzahl korrekte Vorhersagen / Gesamtanzahl)
  - Technisch: Wird durch Vergleich des argmax der Modellausgabe mit den wahren Kategorien-Indizes ermittelt

- **Val ACC**
  - Validierungsgenauigkeit für kategorische Vorhersagen
  - ⬆️ STEIGEND - sollte ähnlich zu Train ACC sein
  - Große Differenz zu Train ACC deutet auf Overfitting hin

## Wichtige Hinweise

- Ein Gleichgewicht zwischen Rekonstruktionsverlusten (MSE/CE) und Regularisierung (KL) ist für ein gutes generatives Modell entscheidend
- Die KL-Divergenz steigt üblicherweise, wenn beta erhöht wird - dies ist Teil des Trainingsdesigns
- Achten Sie auf die Lücke zwischen Training und Validierung - sie sollte nicht zu groß werden