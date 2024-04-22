import flexflow.serve as ff
ff.init(
        num_gpus=1,
        memory_per_gpu=30000,
        zero_copy_memory_per_node=30000,
        tensor_parallelism_degree=1,
        pipeline_parallelism_degree=1
    )

# Specify the LLM
llm = ff.LLM("facebook/opt-6.7b", cache_path="/ocean/projects/cis240042p/ngeorge/.cache/")

# Specify a list of SSMs (just one in this case)
ssms=[]
for i in range(8):
    ssms.append(ff.SSM("JackFram/llama-68m", cache_path="/ocean/projects/cis240042p/ngeorge/.cache/"))
generation_config = ff.GenerationConfig(
    do_sample=False, temperature=0.9, topp=0.8, topk=1
)

for ssm in ssms:
    ssm.compile(generation_config)

# Compile the LLM for inference and load the weights into memory
llm.compile(generation_config,
            max_requests_per_batch = 16,
            max_seq_length = 256,
            max_tokens_per_batch = 128,
            ssms=ssms)

llm.start_server()
result = llm.generate("Here are some travel tips for Tokyo:\n")
llm.stop_server() # This invocation is optional

