# Running fast VLA in Real Time (Pi0 + Pi05)
based on https://github.com/dexmal/realtime-vla.git


```
nsys profile -t cuda,nvtx,osrt -o pi05_pipeline_report python benchmark.py --model_version pi05 --num_views 3 --chunk_size 50 --prompt_len 10
```

 Time (%)  Total Time (ns)  Instances   Avg (ns)    Med (ns)   Min (ns)  Max (ns)  StdDev (ns)   Style                 Range               
 --------  ---------------  ---------  ----------  ----------  --------  --------  -----------  -------  ----------------------------------
     56.5       9395835657        100  93958356.6  93619204.0  93343864  99831034    1064141.8  PushPop  :pi05.pipeline.transformer_encoder
     28.7       4772362898        100  47723629.0  47377438.5  47194725  65154161    1831433.5  PushPop  :pi05.pipeline.transformer_decoder
     14.8       2466714004        100  24667140.0  24650571.5  24388429  25632341     213451.6  PushPop  :pi05.pipeline.vision_encoder 

  