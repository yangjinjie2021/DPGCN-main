a=1
while [ $a -gt 0 ]
do
python dpgcn_main.py --train_file ./data/restaurant_train14.txt --test_file ./data/restaurant_test14.txt --bert_model bert-base-uncased/ --num_epoch 10 --model_name dpgcn --batch_size 16 --learning_rate 1e-5 --outdir ./ --seed 26349 --l2reg 0.001 --pos_hidden 50 --dep_hidden 50 --distance 10 --bert_dropout 0.1
done

