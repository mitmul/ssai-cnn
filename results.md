# Results
## Conventional methods

Model                         | Mass. Buildings | Mass. Roads
:---------------------------- | :-------------- | :----------
MnihCNN                       | 0.9150          | 0.8873
MnihCNN + CRF                 | 0.9211          | 0.8904
MnihCNN + Post-processing net | 0.9203          | 0.9006

## Step = 256, epoch = 100

Model                              | Building-channel | Road-channel
:--------------------------------- | :--------------- | :-----------
ours (multi-channel with MA)       | 0.94015456       | 0.88495376
ours (multi-channel with CIS + MA) | 0.94515369       | 0.89324935

## Step = 256, epoch = 150

Model                              | Building-channel | Road-channel
:--------------------------------- | :--------------- | :-----------
ours (multi-channel)               | 0.94153904       | 0.89100958
ours (multi-channel with MA)       | 0.95132348       | 0.8996919
ours (multi-channel with CIS)      | 0.94224532       | 0.8923019
ours (multi-channel with CIS + MA) | 0.95229794       | 0.90027631

## Step = 256, epoch = 200

Model                              | Building-channel | Road-channel
:--------------------------------- | :--------------- | :-----------
ours (multi-channel)               | 0.938915         | 0.890997
ours (multi-channel with MA)       | 0.9516143        | 0.90166159
ours (multi-channel with CIS)      | 0.940134         | 0.890475
ours (multi-channel with CIS + MA) | 0.95276838       | 0.90142274

## Step = 256, epoch = 250

Model                              | Building-channel | Road-channel
:--------------------------------- | :--------------- | :-----------
ours (multi-channel with MA)       | 0.95279146       | 0.90171033
ours (multi-channel with CIS + MA) | 0.95255047       | 0.9008897

## Step = 256, epoch = 300

Model                              | Building-channel | Road-channel
:--------------------------------- | :--------------- | :-----------
ours (multi-channel with MA)       | 0.952923         | 0.902077
ours (multi-channel with CIS + MA) | 0.951966         | 0.901576

## Step = 256, epoch = 400

Model                              | Building-channel | Road-channel
:--------------------------------- | :--------------- | :-----------
ours (multi-channel with MA)       | 0.95204987       | 0.90253623
ours (multi-channel with CIS + MA) | 0.95251664       | 0.90148289

# Single model
## Step = 256, epoch = 200, dropout

Model                              | Building-channel | Road-channel
:--------------------------------- | :--------------- | :-----------
ours (multi-channel with MA)       | 0.946714         | 0.903493
ours (multi-channel with CIS + MA) | 0.947017         | 0.905389

## Step = 256, epoch = 400, dropout

Model                              | Building-channel | Road-channel
:--------------------------------- | :--------------- | :-----------
ours (multi-channel with MA)       | 0.946714         | 0.903493
ours (multi-channel with CIS + MA) | 0.947017         | 0.905389

# Model averaging
## Step = 256, epoch = 100, dropout

Model                              | Building-channel | Road-channel
:--------------------------------- | :--------------- | :-----------
ours (multi-channel with MA)       | 0.93099659       | 0.88200375
ours (multi-channel with CIS + MA) | 0.93475515       | 0.88366389

## Step = 256, epoch = 200, dropout

Model                              | Building-channel | Road-channel
:--------------------------------- | :--------------- | :-----------
ours (multi-channel with MA)       | 0.95072612       | 0.89861968
ours (multi-channel with CIS + MA) | 0.95098022       | 0.89845107

## Step = 1024, epoch = 300, dropout

Model                              | Building-channel | Road-channel
:--------------------------------- | :--------------- | :-----------
ours (multi-channel with MA)       | 0.95240912       | 0.89946007
ours (multi-channel with CIS + MA) | 0.95276445       | 0.90049365

## Step = 1024, epoch = 400, dropout

Model                              | Building-channel | Road-channel
:--------------------------------- | :--------------- | :-----------
ours (multi-channel with MA)       | 0.95231262       | 0.89971473
ours (multi-channel with CIS + MA) | 0.95280431       | 0.90071099

## CIS Single
result: 0 [ 0.98717177  0.98707403]
result: 1 [ 0.9436488   0.94394724]
result: 2 [ 0.89288454  0.89409248]

result: 0 [ 0.98712131  0.98697123]
result: 1 [ 0.9440021   0.94383582]
result: 2 [ 0.89475327  0.89513849]

result: 0 [ 0.987318    0.98750422]
result: 1 [ 0.9445083   0.94466657]
result: 2 [ 0.89593871  0.8950344 ]

result: 0 [ 0.98690122  0.98707093]
result: 1 [ 0.94411222  0.9446715 ]
result: 2 [ 0.89431934  0.89305437]

result: 0 [ 0.98733737  0.98747745]
result: 1 [ 0.94538692  0.94492537]
result: 2 [ 0.89366111  0.89323461]

result: 0 [ 0.98708133  0.98730423]
result: 1 [ 0.9439836   0.94357188]
result: 2 [ 0.89546564  0.89428484]

result: 0 [ 0.98742897  0.9872073 ]
result: 1 [ 0.94513237  0.94490066]
result: 2 [ 0.89405115  0.8947532 ]

result: 0 [ 0.98672508  0.98669841]
result: 1 [ 0.94283563  0.94284722]
result: 2 [ 0.89385329  0.89366565]

## Multi Single
result: 0 [ 0.98676834  0.98678679]
result: 1 [ 0.943482    0.94298197]
result: 2 [ 0.89253259  0.89349896]

result: 0 [ 0.98671458  0.9868855 ]
result: 1 [ 0.94252649  0.94254146]
result: 2 [ 0.89443769  0.89387986]

result: 0 [ 0.98682528  0.98687003]
result: 1 [ 0.94416111  0.94381609]
result: 2 [ 0.89395184  0.89290773]

result: 0 [ 0.98727091  0.98732937]
result: 1 [ 0.9449592   0.94466241]
result: 2 [ 0.89317871  0.89335852]

result: 0 [ 0.98680073  0.98670082]
result: 1 [ 0.94257691  0.94281478]
result: 2 [ 0.89428897  0.89452913]

result: 0 [ 0.98698049  0.98686956]
result: 1 [ 0.94328121  0.94330325]
result: 2 [ 0.89323064  0.89328006]

result: 0 [ 0.98737076  0.98722732]
result: 1 [ 0.94453414  0.94410367]
result: 2 [ 0.89312458  0.89428682]

result: 0 [ 0.98703347  0.98681946]
result: 1 [ 0.94295965  0.94352485]
result: 2 [ 0.89356337  0.89465463]
