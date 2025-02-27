## FreqNet

Early detection and precise segmentation of colorectal polyps are essential for preventing and treating colorectal cancer. Despite significant advancements in deep learning for colon polyp segmentation, the diverse morphology of polyps and the complexity of endoscopic imaging often lead to models' inability to differentiate polyp boundaries, fine-grained textures, or noise components within high-frequency information. To tackle this challenge, we propose a high-frequency information mining and enhancement network (FreqNET) to achieve robust segmentation. The high-frequency mining module employs learnable nonlinear transformations to adaptively enhance high-frequency features that are beneficial to segmentation results. The high-frequency enhancing module integrates a reverse attention mechanism with an edge feature extraction strategy, significantly improving the completeness of polyp segmentation. We conduct extensive experiments on five benchmark datasets, with FreqNET achieving state-of-the-art (SOTA) performance on Kvasir-SEG and ClinicDB. For the more challenging ETIS dataset, FreqNET improved the key metrics mDice and mIOU by 1\% and 1.6\%, respectively. Qualitative analyses further demonstrate that FreqNET maintains accurate segmentation performance even when confronted with complex endoscopic images that are difficult for existing models to process.


### Proposed Baseline
#### Training/Testing
The training and testing experiments are conducted using PyTorch with a single NVIDIA 3090 with 24 GB Memory.<br>
Note that our model also supports low memory GPU, which means you can lower the batch size.<br>
downloading testing dataset and move it into `./data/TestDataset/`, which can be found in this [download link (327.2MB)](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view). It contains five sub-datsets: CVC-300 (60 test samples), CVC-ClinicDB (62 test samples), CVC-ColonDB (380 test samples), ETIS-LaribPolypDB (196 test samples), Kvasir (100 test samples).<br>
downloading training dataset and move it into `./data/TrainDataset/`, which can be found in this [download link (399.5MB)](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view). It contains two sub-datasets: Kvasir-SEG (900 train samples) and CVC-ClinicDB (550 train samples).

downloading pretrained weights and move it into ./checkpoint/, which can be found in this [download link](https://pan.baidu.com/s/13Pv8xEGNAM3KxJgT-mLwZw?pwd=u5rd).

Say what the step will be

/```
Give the example
/```

And repeat

/```
until finished
/```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

/```
Give an example
/```

### And coding style tests

Explain what these tests test and why

/```
Give an example
/```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

