#!/bin/sh

mkdir -p datasets/custom

pushd datasets/custom
curl -L 'https://upload.wikimedia.org/wikipedia/commons/d/d7/C172_Cessna_Skyhawk_PH-HBW_at_Teuge_07March2009.JPG' -o airplane.jpg
curl -L 'https://upload.wikimedia.org/wikipedia/commons/a/a4/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg' -o automobile.jpg
curl -L 'https://upload.wikimedia.org/wikipedia/commons/d/d5/Grey_Heron._AMSM4086.jpg' -o bird.jpg
curl -L 'https://upload.wikimedia.org/wikipedia/commons/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg' -o cat.jpg
curl -L 'https://upload.wikimedia.org/wikipedia/commons/b/b7/White-tailed_deer.jpg' -o deer.jpg
curl -L 'https://upload.wikimedia.org/wikipedia/commons/d/d5/Retriever_in_water.jpg' -o dog.jpg
curl -L 'https://upload.wikimedia.org/wikipedia/commons/6/68/Wood_Frog_%28Rana_sylvatica%29_%2825234151669%29.jpg' -o frog.jpg
curl -L 'https://upload.wikimedia.org/wikipedia/commons/d/d6/Przewalski%27s_Horse_at_The_Wilds.jpg' -o horse.jpg
curl -L 'https://upload.wikimedia.org/wikipedia/commons/1/12/Albatun_Dod.jpg' -o ship.jpg
curl -L 'https://upload.wikimedia.org/wikipedia/commons/0/06/Walmart%E2%80%99s_Hybrid_Assist_Truck.jpg' -o truck.jpg
popd
