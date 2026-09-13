# Do Fly Brains *actually* think in MORGAN style spectral MoE??? Probably not, but here is what I did find...
### Testing whether a biologically grounded graph structure can serve useful for a model that is simultaneously efficient, interpretable, and non-linear.
## Why..?
While waiting on my SWAEV AI model training runs to finish I've been scrolling and been seeing a lot of goofy projects related to mapping the fly brain to trade stock, or importing it to Minecraft and training it on those tasks. Resulting in a digital fly doing silly things. Seeing a fly play beat saber made me question what the hell is this all about? Are people using the actual biophysical neurological replica of a fly brain and forcing it to learn these novel tasks... or maybe the answer is more simpler. I have found that for most of these project it is in fact more boring than it seems :(

My understanding is this all stems from researchers that mapped out all the pathways in a fly's brain. Huge respect. That is awesome and the dedication is insane. Releasing this in form of a graph of all the connections is the basis that these projects go off of. Mapping perceptrons onto the nodes of the graph in place of biological neurons is the methods they are using to bring the brain "to life". So the architecture of the graph serving as an actual fly in these projects is kinda a stretch to me. I mean its just altering the structure of a neural network... the rest is just standard dense neural network and transformer practice. This is true at least for these goofy applications of the fly brain structure. I have found that there are researchers genuinely creating novel neural network architectures to mimic and capture the biophysicalities of the fly. Sick. I love applying biophysics to neural networks to make them make more sense and learn more about biology. Here are some cool stuff I did find:

+ **FlyGM (Fly-connectomic Graph Model)** -> *Zehao Jin, Yaoye Zhu, Chen Zhang, Yanan Sui* (**Tsinghua / Georgia Tech**). Directed message-passing graph where **nodes are neurons** partitioned into *afferent / intrinsic / efferent pools* and **edges encode real FlyWire connectivity**. Synapse weights define a *sparse linear aggregation operator*; each neuron carries trainable intrinsic parameters. Trained with **PPO** for whole-body locomotion.

+ **BPU (Biological Processing Unit)** -> *Siyu Yu, Zihan Qin, Tingshan Liu, Beiya Xu, R. Jacob Vogelstein, Jason Brown, Joshua T. Vogelstein* (**Johns Hopkins**). Fixed-weight recurrent core from the complete *Drosophila larval connectome* (~3,000 neurons, ~65k synapses). Synaptic weights are taken directly from the connectome and **never updated**; only input and output projections are trained. Scaled via *degree-corrected stochastic block model*.

+ **Shiu et al. LIF model** -> *Philip Shiu, Gabriella Sterne, Salil Bidaye* (**UC Berkeley → Eon Systems**). Whole-brain leaky integrate-and-fire spiking network built from the adult **FlyWire connectome** (~140k neurons, ~50M synapses). **No training at all** — connectivity + predicted neurotransmitter identity alone predict motor output (**95% accuracy** on proboscis extension).

+ **Lappalainen et al. connectome-constrained network** -> *Janne K. Lappalainen, Fabian D. Tschopp, Sridhama Prakhya, Mason McGill, Aljoscha Nern, Kazunori Shinomiya, Shin-ya Takemura, Eyal Gruntman, Jakob H. Macke, Srinivas C. Turaga* (**Tübingen / Janelia / Stanford**). Recurrent network for **64 visual cell types** where connectivity structure is fixed from the connectome; only a reduced set of parameters is learned via a *motion-detection task*. Predicts neural activity across the visual system.

+ **KCNet** -> *Jinyung Hong, Theodore P. Pavlic* (**Arizona State**). Single-hidden-layer network with **sparse, randomized binary weights** that mimic the high-dimensional *Kenyon Cell representation* in the fly olfactory system. Uses **straight-through gradient estimation** for dynamic weight exploration.

+ **Eon Systems embodied fly** -> *Eon Systems team* (building on *Shiu et al.* + *Lappalainen et al.* + *NeuroMechFly*). Full whole-brain **LIF model** coupled to a biomechanical body in simulation; the first case of a **connectome-derived brain driving a virtual embodied organism**.

*After this dive I decided to add in my own efforts. I am bored and my curiousity had spiked.*

## My approach.
So the source of it all is the FlyWire connectome. It is essentially a complete wiring diagram of an adult Drosophila brain (YAY I worked on fruitfly dna before!). Entailing "roughly 140,000 neurons and 50 million synaptic connections". Neurons as nodes and synaptic connections just refers to the wires between these nodes.

*Now that I have done the due diligence of prerequisite research, Time to start applying my domain.*

MoE design is the non-negotiable for me... I mean it's an absolute fact that different parts of our brains specialise in certain actions. MoE LLMs have been seen to be lightweight in inference and also have superb performance. For a biology inspired NN this makes sense. Off course we are working with a graph of a fly's brain, so the build starts off with a directed graph **G** for **G** **=** **(V,E)**. Now **V** is the set of neurons in the brain and **E** is the set of synaptic connections in said neurons.


