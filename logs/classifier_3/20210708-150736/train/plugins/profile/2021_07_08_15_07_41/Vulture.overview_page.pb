�	�:� @@�:� @@!�:� @@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�:� @@�!H��?1Ͼ� =%=@Ac('�UH�?I�h[ͺ�?rEagerKernelExecute 0*	cX9��M@2s
<Iterator::Model::MaxIntraOpParallelism::MapAndBatch::Shuffle %�����?!Z�%���F@)%�����?1Z�%���F@:Preprocessing2j
3Iterator::Model::MaxIntraOpParallelism::MapAndBatchv���z�?!�h��68@)v���z�?1�h��68@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismB!�J�?!���ٔH@)��!��?1I��,�7@:Preprocessing2F
Iterator::Model�~j�t��?!�a�db;K@)��VC�n?1+�Xl^@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 4.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�4.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI �xQ�!@Q\����V@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�!H��?�!H��?!�!H��?      ��!       "	Ͼ� =%=@Ͼ� =%=@!Ͼ� =%=@*      ��!       2	c('�UH�?c('�UH�?!c('�UH�?:	�h[ͺ�?�h[ͺ�?!�h[ͺ�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q �xQ�!@y\����V@�"g
;gradient_tape/pointnet/conv1d_7/conv1d/Conv2DBackpropFilterConv2DBackpropFilterkD��?!kD��?0"g
;gradient_tape/pointnet/conv1d_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter8�꿖�?!��z�kɠ?0"0
Adam/gradients/AddN_18AddN�;����?!w�z̮��?"0
Adam/gradients/AddN_34AddN���8��?!�����G�?":
!pointnet/global_max_pooling1d/MaxMax�z[j��?!-9��� �?"<
#pointnet/global_max_pooling1d_1/MaxMax=e�N�?!�ŀd��?"<
#pointnet/global_max_pooling1d_2/MaxMax�9�Av�?!��X�?"E
+pointnet/batch_normalization_2/moments/meanMean���ŵ�?!0�1@O�?"[
Bgradient_tape/pointnet/batch_normalization_2/batchnorm/add_1/Sum_1Sumn�7I��?!�Y,g�ƽ?"[
Bgradient_tape/pointnet/batch_normalization_2/batchnorm/mul_1/Sum_1Sumn�7I��?!ۃ��R�?Q      Y@Y�DX��,@ak��t�jU@q����@y��u��|?"�

both�Your program is POTENTIALLY input-bound because 4.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�4.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Ampere)(: B 