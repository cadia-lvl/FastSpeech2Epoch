train_file = '/work/shijun/ljspeech/epoch_processed2/train.txt'
val_file = '/work/shijun/ljspeech/epoch_processed2/val.txt'

symbols = []

with open(train_file) as fp:
    for line in fp:
        line = line.strip().split('|')
        phonmes = line[2].split(' ')
        
        symbols.extend(phonmes)
        
symbols = set(list(symbols))
print(' '.join(symbols))

with open(val_file) as fp:
    for line in fp:
        line = line.strip().split('|')
        phonmes = line[2].split(' ')
        
        for p in phonmes:
            if p not in symbols:
                print(p)
                exit()