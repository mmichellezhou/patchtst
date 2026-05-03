# Data

Download the ETT datasets by running the following command from the `data/` directory:

```bash
git clone https://github.com/zhouhaoyi/ETDataset.git
```

The following files are used in our experiments:
- `ETDataset/ETT-small/ETTh1.csv`
- `ETDataset/ETT-small/ETTh2.csv`
- `ETDataset/ETT-small/ETTm1.csv`
- `ETDataset/ETT-small/ETTm2.csv`

Download electricity dataset:
```bash
curl -L https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/electricity/electricity.csv -o electricity.csv
```

Download weather dataset:
```bash
curl -L https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv -o weather.csv
```

Download traffic dataset:
```bash
curl -L https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/traffic/traffic.csv -o traffic.csv
```

Download ILI (national illness) dataset:
```bash
curl -L https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/illness/national_illness.csv -o national_illness.csv
```