# Gaussian Flow Bridges for audio domain transfer with unpaired data

Audio domain transfer is the process of modifying audio signals to match characteristics of a different domain, while retaining the original content. This paper investigates the potential of Gaussian Flow Bridges, an emerging approach in generative modeling, for this problem. The presented framework addresses the transport problem across different distributions of audio signals through the implementation of a series of two deterministic probability flows. The proposed framework facilitates manipulation of the target distribution properties through a continuous control variable, which defines a certain aspect of the target domain. Notably, this approach does not rely on paired examples for training. To address identified challenges on maintaining the speech content consistent, we recommend a training strategy that incorporates chunk-based minibatch Optimal Transport couplings of data samples and noise. Comparing our unsupervised method with established baselines, we find competitive performance in tasks of reverberation and distortion manipulation.Despite encoutering limitations, the intriguing results obtained in this study underscore potential for further exploration.

![](assets/diagram.png)

In this repository, we provide the sample code to train a Gaussian Flow Bridge (GFB) for controlling speech reverberation or clipping, presented in the paper ["Gaussian Flow Bridges for audio domain transfer with unpaired data"](https://) submitted to IEEE IWAENC 2024. 
We hope this sample code enables reproducibility of our proposed method and results and invites further work on the topic of audio domain transfer.
Audio examples are available at [https://microsoft.github.io/GFB-audio-control/dist](https://microsoft.github.io/GFB-audio-control/dist) 

## Paper
If you use this code in your research please cite the following [publication](https://):
```
@inproceedings{emoliner2024,
  title={Gaussian Flow Bridges for audio domain transfer with unpaired data},
  author={Moliner, Eloi and Braun, Sebastian and Gamper, Hannes},
  journal={arxiv},
  notes={Submitted to IEEE IWAENC 2024}
}
```

This paper can also be found on arXiv at [...](...).

-----
[LICENSE](https://github.com/microsoft/GFB-audio-control/blob/master/LICENSE)


[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)


[Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkId=521839)

-----

## Responsible AI Transparency Information 

An AI system includes not only the technology, but also the people who will use it, the people who will be affected by it, and the environment in which it is deployed. Creating a system that is fit for its intended purpose requires an understanding of how the technology works, its capabilities and limitations, and how to achieve the best performance. Microsoft has a broad effort to put our AI principles into practice. To find out more, see [Responsible AI principles from Microsoft](https://www.microsoft.com/en-us/ai/responsible-ai). 

### Use of this code 

The purpose of this sample code is to enable reproducibility of our method and results and to encourage future work on the topic. 

### Project data 

The companion [website](https://microsoft.github.io/GFB-audio-control/dist) contains a selection of audio samples illustrating the performance of the proposed model. The samples are taken from the publicly available [DAPS dataset](https://zenodo.org/records/4660670), and [GuitarSet](https://zenodo.org/records/3371780).

### Fairness and Responsible AI testing 

At Microsoft, we strive to empower every person on the planet to do more. An essential part of this goal is working to create technologies and products that are fair and inclusive. Fairness is a multi-dimensional, sociotechnical topic and impacts many different aspects of our work.  

When systems are deployed, Responsible AI testing should be performed to ensure safe and fair operation for the specific use case. No Responsible AI testing has been done to evaluate this method including validating fair outcomes across different groups of people. Responsible AI testing should be done before using this code in any production scenario. 

> Note: The documentation included in this ReadMe file is for informational purposes only and is not intended to supersede the applicable license terms. 

### Limitations

The proposed model was trained and evaluated on clean recordings of English speech. It may fail for other languages or acoustic conditions without retraining. Even for English speech, the model may occasionally introduce acoustic or semantic distortions, or alter the perceived speaker identity. For input other than noise-free English speech, the model behaviour is undefined.

The code published in this repository is intended purely to illustrate the methods introduced in the accompanying publication. 

## Contributing

This code has been created as part of a summer intern project by Eloi Moliner (eloi.moliner@aalto.fi), Ph.D. Student, Aalto University, Finland

---

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Requirements

We recommend using the [Anaconda](https://docs.anaconda.com/anaconda/install/) distribution for python dependencies. The code should work will any recent version, but it was tested with Python 3.9.19. The required packages are listed in `requirements.txt`, and can be installed by running the following command:


```bash
pip install -e .
pip install -r requirements.txt 
```


## How to use

### Training 

To train the models, follow these steps:

1. Integrate your dataset by modifying `datasets/datasets.py` and adding a configuration file in `conf/datasets/new_dataset.yaml`.
2. Run the training scripts for the experiments reported in the paper:

```bash
# Speech reverberation
python src/train.py --config-name=conf_speech_reverb.yaml

# Speech clipping
python src/train.py --config-name=conf_speech_clipping.yaml
```


---

### Inference 

To perform inference, adjust the test parameters by adding a new tester configuration file in `conf/tester/{}.yaml` or modifying an existing one. Then, run the inference as follows:

```bash
python src/test.py --config-name=conf_speech_reverb.yaml --tester=reverb_bridge.yaml --checkpoint=$checkpoint_filename
```

### Trained models

We provide the checkpoints corresponding to the models evaluated in the paper. They can be downloaded as follows:

```bash
wget $ckpt_url #Model trained on reverberant speech using Conditional Flow Matching and independent couplings
wget $ckpt_url #Model trained on reverberant speech using Conditional Flow Matching and Chunked-OT couplings with chunk size Nc=512
wget $ckpt_url #Model trained on reverberant speech using Conditional Flow Matching and Chunked-OT couplings with chunk size Nc=256
wget $ckpt_url #Model trained on reverberant speech using Conditional Flow Matching and Chunked-OT couplings with chunk size Nc=128
wget $ckpt_url #Model trained on clipped (and clean) speech using Conditional Flow Matching and independent couplings
wget $ckpt_url #Model trained on clipped (and clean) speech using Conditional Flow Matching and Chunked-OT couplings with chunk size Nc=512
```


-----


