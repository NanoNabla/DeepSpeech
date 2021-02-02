Project DeepSpeech with Horovod distributed training
====================================================
This branch contains an implementation for distributed training in DeepSpeech using `Horovod framework <https://github.com/horovod/horovod>`_.

DeepSpeech is an open-source Speech-To-Text engine, using a model trained by machine learning techniques based on `Baidu's Deep Speech research paper <https://arxiv.org/abs/1412.5567>`_. Project DeepSpeech uses Google's `TensorFlow <https://www.tensorflow.org/>`_ to make the implementation easier.

This branch should be compatible with Mozilla's DeepSpeech. Use distributed training using horovod by

.. code-block::

    horovodrun -np 2 python ../DeepSpeech.py --horovod True [...]

Documentation for installation, usage, and training models are available on `deepspeech.readthedocs.io <https://deepspeech.readthedocs.io/?badge=latest>`_ and `github.com/horovod/horovod <https://github.com/horovod/horovod>`_.


