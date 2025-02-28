�	��c�H@��c�H@!��c�H@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC��c�H@{��{�
�?1���eG@A��~��@�?I͓k
d��?rEagerKernelExecute 0*	�n��jR@2s
<Iterator::Model::MaxIntraOpParallelism::MapAndBatch::Shuffle c~nh�N�?!�nWN��N@)c~nh�N�?1�nWN��N@:Preprocessing2j
3Iterator::Model::MaxIntraOpParallelism::MapAndBatch߿yq⫍?!c����3@)߿yq⫍?1c����3@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism���̓k�?!�*��A@)6=((E+�?1>�B��.@:Preprocessing2F
Iterator::Model�/�
Ҝ?!W���CC@)�8�j�3c?1a�jt	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 3.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI�PV6j@QD���\iW@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	{��{�
�?{��{�
�?!{��{�
�?      ��!       "	���eG@���eG@!���eG@*      ��!       2	��~��@�?��~��@�?!��~��@�?:	͓k
d��?͓k
d��?!͓k
d��?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�PV6j@yD���\iW@�"g
;gradient_tape/pointnet/conv1d_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter�s	�?!�s	�?0"g
;gradient_tape/pointnet/conv1d_7/conv1d/Conv2DBackpropFilterConv2DBackpropFilter������?!���ZF��?0"0
Adam/gradients/AddN_18AddN�:L��?!b�倐<�?"0
Adam/gradients/AddN_34AddN>0�� �?!@FDxޱ?"R
5gradient_tape/pointnet/global_max_pooling1d_1/truedivRealDiv��%]%��?!���o�?"P
3gradient_tape/pointnet/global_max_pooling1d/truedivRealDivS��y}�?!�V�.L��?"e
:gradient_tape/pointnet/conv1d_2/conv1d/Conv2DBackpropInputConv2DBackpropInputV��� C�?!��M���?0"e
:gradient_tape/pointnet/conv1d_7/conv1d/Conv2DBackpropInputConv2DBackpropInput�Jגu7�?!5H�O�?0"J
,gradient_tape/pointnet/activation_9/ReluGradReluGrad�p�?!�I$p��?"J
,gradient_tape/pointnet/activation_2/ReluGradReluGradh�Y���?!��y-^�?Q      Y@Y��wm��,@a	Q�,lU@q���qO�@y-��q�;r?"�	
both�Your program is POTENTIALLY input-bound because 3.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Ampere)(: B 