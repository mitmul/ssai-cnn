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
