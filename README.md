# Knowledge Distillation Compression

Compression of models ensemble using Knowledge Distillation. Distillation of multiple models built using Bootstrap
Aggregation into one single network for efficient deployment. Network trained using a combination of the Kullback–Leibler
divergence with respect to the teachers results after averaging, and the Cross-Entropy with respect to the hard ground-truth labels.
Built on [PyTorch](https://pytorch.org/).

## Usage

```shell
$ conda create --name KDCompression --file requirements.txt
```

## System Requirements

Python 3.9.2

[PyTorch](https://pytorch.org/) 1.8.1

Tested with RTX2080Ti and Intel i9-10900k

## References

*[1] G. Hinton et al. “Distilling the Knowledge in a Neural Network.”*

*[2] Z. Allen-Zhu et al., “Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning.”*

*[3] S. Park et al. “On the Orthogonality of Knowledge Distillation with Other Techniques: From an Ensemble Perspective.”*

## Contacts

For any inquiries please contact: 
[Alessandro Delmonte](https://aledelmo.github.io) @ [alessandro.delmonte@institutimagine.org](mailto:alessandro.delmonte@institutimagine.org)

## License

This project is licensed under the [Apache License 2.0](LICENSE) - see the [LICENSE](LICENSE) file for
details