# BACPI

[BACPI](https://academic.oup.com/bioinformatics/article-abstract/38/7/1995/6511437): a bi-directional attention neural network for compound-protein interaction and binding affinity prediction

## Requirements

[numpy](https://numpy.org/)==1.19.1

[pandas](https://pandas.pydata.org/)==1.0.5

[rdkit](https://www.rdkit.org/)==2009.Q1-1

[scikit_learn](https://scikit-learn.org/stable/)==1.0.2

[scipy](https://scipy.org/)==1.5.0

[torch](https://pytorch.org/)==1.6.0

## Example usage

### 1. Compound–protein interaction prediction
This is a binary classification task, which aim to predict whether there is an interaction between the compound and the protein or not.
```bash
# Run the commandline
python main.py -task interaction -dataset human

```

### 2. Compound–protein binding affinity prediction
This is a regression task, which aim to predict a continuous value named binding affinity that reflects how tightly the compound binds to a particular target protein.
```bash
# Run the commandline
python main.py -task affinity -dataset Kd

```

### 3. Specify hyperparameters
There are many hyperparameters of the model (i.g. learning rate, batch size, epochs, step size, weight decay, hidden size, window size, layer size). For more information, see source code of main.py.
 ```bash
# Run the commandline
python main.py -task affinity -dataset Kd -lr 0.001 -batch_size 64 -num_epochs 15

```

### 4. Run on your datasets
**NOTICE : Your dataset should be divided into train and test sets**

Please prepare train.txt and test.txt file, and store them in the corresponding paths ('./data/interaction/dataset_name/' for CPI prediction and './data/affinity/dataset_name/' for affinity prediction). Each line in the data file contains the SMILES of the compound, the amino acid sequence of the protein, and the predicted label (interaction label: 0 or 1, affinity label: a continuous value), separated by commas(,). The format of the data file is as follows: 
 ```bash
# interaction dataset
CC(C)C1=NN2C=CC=CC2=C1C(=O)C(C)C,MVDEDKKSGTRVFKKTSPNGKITTYLGKRDFIDRGDYVDLIDGMVLIDEEYIKDNRKVTAHLLAAFRYGREDLDVLGLTFRKDLISETFQVYPQTDKSISRPLSRLQERLKRKLGANAFPFWFEVAPKSASSVTLQPAPGDTGKPCGVDYELKTFVAVTDGSSGEKPKKSALSNTVRLAIRKLTYAPFESRPQPMVDVSKYFMMSSGLLHMEVSLDKEMYYHGESISVNVHIQNNSNKTVKKLKIYIIQVADICLFTTASYSCEVARIESNEGFPVGPGGTLSKVFAVCPLLSNNKDKRGLALDGQLKHEDTNLASSTILDSKTSKESLGIVVQYRVKVRAVLGPLNGELFAELPFTLTHSKPPESPERTDRGLPSIEATNGSEPVDIDLIQLHEELEPRYDDDLIFEDFARMRLHGNDSEDQPSPSANLPPSLL,0
C1=CC=C2C(=C1)N=C(S2)C(C#N)C3=NC(=NC=C3)NCCC4=CN=CC=C4,MFRQEILNEVLFIVPNRYVDLLPSQFGNAMEVIAFDQISERRVVIKKVVLPENFDNWQHWRRAQRELFCTLHIQEENFVKMYSIYTWVETVEEMREFYTVREYMDWNLRNFILSTPEKLDHKVIKSIFFDVCLAVQYMHSIRVGHRDLKPENVLINYEAIAKISGFAHANREDPFVNTPYIVQRFYRAPEILCETMDNNKPSVDIWSLGCILAELLTGKILFTGQTQIDQFFQIVRFLGNPDLSFYMQMPDSARTFFLGLPMNQYQKPTNIHEHFPNSLFLDTMISEPIDCDLARDLLFRMLVINPDDRIDIQKILVHPYLEEVWSNIVIDNKIEEKYPPIALRRFFEFQAFSPPRQMKDEIFSTLTEFGQQYNIFNNSRN,1
...

# affinity dataset
COC1=CC=C(C=C1)CNS(=O)(=O)C2=CC=C(S2)S(=O)(=O)N,MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK,9.309803919971486
C1=CSC(=C1)CNS(=O)(=O)C2=CC=C(S2)S(=O)(=O)N,MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK,9.080921907623926
...

```
