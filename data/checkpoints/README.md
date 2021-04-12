
# CHECKPOINTS

### Manual Downloads
Individual model checkpoints are available at this [google drive folder](https://drive.google.com/drive/folders/1jgNpYe4TVSF4mMBVtFh4QfB2GovNPdh7?usp=sharing)

### Automated Downloads

```python
import neuspell
print(neuspell.seq_modeling.downloads.CHECKPOINTS_NAMES)
```
```python
neuspell.seq_modeling.downloads.download_pretrained_model("subwordbert-probwordnoise")
```