# Imputing fehlender Werte mit TabSyn

TabSyn nutzt eine innovative Kombination aus Variational Autoencodern (VAE) und Diffusionsmodellen, um fehlende Werte in tabellarischen Daten präzise zu ergänzen. Dieser Prozess wird als "Imputing" bezeichnet und ermöglicht eine datengetreue Vervollständigung unvollständiger Datensätze.

## Grundprinzip

Der Imputing-Prozess in TabSyn unterscheidet sich von herkömmlichen Methoden durch:

1. **Latente Raumtransformation**: Statt direkt mit rohen Daten zu arbeiten, werden diese zunächst in einen latenten Raum transformiert
2. **Diffusionsbasierte Generierung**: Fehlende Werte werden durch einen kontrollierten Diffusionsprozess generiert
3. **Konditionierung auf bekannte Werte**: Vorhandene Werte bleiben erhalten und steuern die Generierung fehlender Werte

## Funktionsweise in 5 Schritten

### 1. Datenvorbereitung
- Laden der Daten und Definition der zu imputierenden Spalten
- Temporäre Ersetzung fehlender Werte durch einfache Statistiken (Mittelwert/häufigste Klasse)
- Trennung in numerische und kategorische Features

### 2. Latente Repräsentation mit VAE
- Transformation der Daten in einen kompakten latenten Raum durch einen vortrainierten VAE
- Normalisierung der latenten Darstellungen für stabileres Training
- Erstellung einer Maske, die fehlende Werte im latenten Raum identifiziert

### 3. Diffusionsprozess initialisieren
- Starts mit zufälligem Rauschen für die zu imputierenden Werte
- Definition einer Sequenz abnehmender Rauschstärken (von σ_max bis σ_min)
- Vorbereitung des vortrainierten Diffusionsmodells für den Entrauschungsprozess

### 4. Schrittweise Entrauschung mit Konditionierung
- Iterative Anwendung des Diffusionsmodells zur Rauschentfernung
- Fixierung bekannter Werte während des gesamten Prozesses
- Mehrfache Iterationen pro Rauschstufe für stabilere Ergebnisse
- Kombination generierter und bekannter Werte gemäß der definierten Maske
```
# Initialisierung der Diffusionsparameter
num_steps = 50  # Anzahl der Entrauschungsschritte
N = 20          # Anzahl der Iterationen pro Schritt
net = model.denoise_fn_D

# Initialisierung mit Rauschen
num_samples, dim = x.shape[0], x.shape[1]
x_t = torch.randn([num_samples, dim], device='cuda')

# Definition der Rauschstärken-Sequenz (von hoch nach niedrig)
step_indices = torch.arange(num_steps, dtype=torch.float32, device=x_t.device)
sigma_min = max(SIGMA_MIN, net.sigma_min)
sigma_max = min(SIGMA_MAX, net.sigma_max)
t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

# Vorbereitung für den Entrauschungsprozess
mask = mask.to(torch.int).to(device)
x_t = x_t.to(torch.float32) * t_steps[0]

# Hauptschleife des Entrauschungsprozesses
with torch.no_grad():
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        print(i)  # Fortschrittsanzeige
        if i < num_steps - 1:
            for j in range(N):
                # Aktuelle und nächste Rauschstufe für bekannte Werte
                n_curr = torch.randn_like(x).to(device) * t_cur
                n_prev = torch.randn_like(x).to(device) * t_next

                # Bekannte Werte mit kontrolliertem Rauschen
                x_known_t_prev = x + n_prev
                
                # Entrauschung für unbekannte Werte
                x_unknown_t_prev = step(net, num_steps, i, t_cur, t_next, x_t)
                
                # Kombination bekannter und unbekannter Werte gemäß der Maske
                x_t_prev = x_known_t_prev * (1-mask) + x_unknown_t_prev * mask

                # Zufälliges Rauschen für die nächste Iteration (außer bei der letzten)
                n = torch.randn_like(x) * (t_cur.pow(2) - t_next.pow(2)).sqrt()

                if j == N - 1:
                    x_t = x_t_prev                   # Letzte Iteration: kein zusätzliches Rauschen
                else:
                    x_t = x_t_prev + n               # Sonst: neues Rauschen für die nächste Iteration

```

### 5. Rekonstruktion und Nachbearbeitung
- Transformation der entrauschten latenten Darstellung zurück in den Originalraum
- Anwendung inverser Normalisierungen und Transformationen
- Konvertierung numerischer Vorhersagen in entsprechende Datentypen
- Umwandlung kategorischer Logits in diskrete Klassen

## Mathematische Grundlage

Der Imputing-Prozess basiert auf dem Bayes'schen Prinzip:

P(x_fehlend | x_bekannt) ∝ P(x_bekannt | x_fehlend) × P(x_fehlend)

Wobei:
- P(x_fehlend | x_bekannt): Wahrscheinlichkeit der fehlenden Werte, gegeben die bekannten Werte
- P(x_bekannt | x_fehlend): Likelihood-Funktion
- P(x_fehlend): Prior-Annahme über fehlende Werte

Das Diffusionsmodell approximiert diesen Prozess durch schrittweise Entrauschung und konditionierte Generierung.

## Vorteile gegenüber traditionellen Methoden

- **Komplexe Abhängigkeiten**: Erfasst nichtlineare Beziehungen zwischen Variablen
- **Multimodale Verteilungen**: Kann mehrere plausible Imputationen für denselben Kontext generieren
- **Konsistenz**: Erhält die statistischen Eigenschaften und Korrelationen des Originaldatensatzes
- **Kategorische Variablen**: Effektive Behandlung kategorialer Daten ohne One-Hot-Encoding-Probleme
- **Unsicherheitsquantifizierung**: Möglichkeit, mehrere Imputationen zu erzeugen und die Variabilität zu analysieren

## Nutzung

Die Imputation wird mit dem bereitgestellten Skript durchgeführt, wobei folgende Parameter angegeben werden können:
- `--dataname`: Name des zu bearbeitenden Datensatzes
- `--gpu`: Zu verwendende GPU (oder -1 für CPU)

Nach Abschluss werden die imputierten Daten im Verzeichnis `impute/{dataname}/` gespeichert.