 ë
Ê
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68¥ë

Playlists_Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_namePlaylists_Dense_1/kernel

,Playlists_Dense_1/kernel/Read/ReadVariableOpReadVariableOpPlaylists_Dense_1/kernel* 
_output_shapes
:
*
dtype0

Playlists_Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namePlaylists_Dense_1/bias
~
*Playlists_Dense_1/bias/Read/ReadVariableOpReadVariableOpPlaylists_Dense_1/bias*
_output_shapes	
:*
dtype0

Playlists_Dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_namePlaylists_Dense_2/kernel

,Playlists_Dense_2/kernel/Read/ReadVariableOpReadVariableOpPlaylists_Dense_2/kernel* 
_output_shapes
:
*
dtype0

Playlists_Dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namePlaylists_Dense_2/bias
~
*Playlists_Dense_2/bias/Read/ReadVariableOpReadVariableOpPlaylists_Dense_2/bias*
_output_shapes	
:*
dtype0

Playlists_Dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_namePlaylists_Dense_3/kernel

,Playlists_Dense_3/kernel/Read/ReadVariableOpReadVariableOpPlaylists_Dense_3/kernel* 
_output_shapes
:
*
dtype0

Playlists_Dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namePlaylists_Dense_3/bias
~
*Playlists_Dense_3/bias/Read/ReadVariableOpReadVariableOpPlaylists_Dense_3/bias*
_output_shapes	
:*
dtype0

Tracks_Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameTracks_Dense_1/kernel

)Tracks_Dense_1/kernel/Read/ReadVariableOpReadVariableOpTracks_Dense_1/kernel*
_output_shapes
:	*
dtype0

Tracks_Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameTracks_Dense_1/bias
x
'Tracks_Dense_1/bias/Read/ReadVariableOpReadVariableOpTracks_Dense_1/bias*
_output_shapes	
:*
dtype0

Playlists_Latent/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*(
shared_namePlaylists_Latent/kernel

+Playlists_Latent/kernel/Read/ReadVariableOpReadVariableOpPlaylists_Latent/kernel*
_output_shapes
:	@*
dtype0

Playlists_Latent/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_namePlaylists_Latent/bias
{
)Playlists_Latent/bias/Read/ReadVariableOpReadVariableOpPlaylists_Latent/bias*
_output_shapes
:@*
dtype0

Tracks_Latent/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*%
shared_nameTracks_Latent/kernel
~
(Tracks_Latent/kernel/Read/ReadVariableOpReadVariableOpTracks_Latent/kernel*
_output_shapes
:	@*
dtype0
|
Tracks_Latent/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameTracks_Latent/bias
u
&Tracks_Latent/bias/Read/ReadVariableOpReadVariableOpTracks_Latent/bias*
_output_shapes
:@*
dtype0

Prediction_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_namePrediction_1/kernel
}
'Prediction_1/kernel/Read/ReadVariableOpReadVariableOpPrediction_1/kernel* 
_output_shapes
:
*
dtype0
{
Prediction_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namePrediction_1/bias
t
%Prediction_1/bias/Read/ReadVariableOpReadVariableOpPrediction_1/bias*
_output_shapes	
:*
dtype0

Prediction_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_namePrediction_2/kernel
}
'Prediction_2/kernel/Read/ReadVariableOpReadVariableOpPrediction_2/kernel* 
_output_shapes
:
*
dtype0
{
Prediction_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namePrediction_2/bias
t
%Prediction_2/bias/Read/ReadVariableOpReadVariableOpPrediction_2/bias*
_output_shapes	
:*
dtype0

Prediction_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_namePrediction_3/kernel
}
'Prediction_3/kernel/Read/ReadVariableOpReadVariableOpPrediction_3/kernel* 
_output_shapes
:
*
dtype0
{
Prediction_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namePrediction_3/bias
t
%Prediction_3/bias/Read/ReadVariableOpReadVariableOpPrediction_3/bias*
_output_shapes	
:*
dtype0

Prediction_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_namePrediction_4/kernel
}
'Prediction_4/kernel/Read/ReadVariableOpReadVariableOpPrediction_4/kernel* 
_output_shapes
:
*
dtype0
{
Prediction_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namePrediction_4/bias
t
%Prediction_4/bias/Read/ReadVariableOpReadVariableOpPrediction_4/bias*
_output_shapes	
:*
dtype0

Final_Prediction/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameFinal_Prediction/kernel

+Final_Prediction/kernel/Read/ReadVariableOpReadVariableOpFinal_Prediction/kernel*
_output_shapes
:	*
dtype0

Final_Prediction/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameFinal_Prediction/bias
{
)Final_Prediction/bias/Read/ReadVariableOpReadVariableOpFinal_Prediction/bias*
_output_shapes
:*
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

 Nadam/Playlists_Dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Nadam/Playlists_Dense_1/kernel/m

4Nadam/Playlists_Dense_1/kernel/m/Read/ReadVariableOpReadVariableOp Nadam/Playlists_Dense_1/kernel/m* 
_output_shapes
:
*
dtype0

Nadam/Playlists_Dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Nadam/Playlists_Dense_1/bias/m

2Nadam/Playlists_Dense_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/Playlists_Dense_1/bias/m*
_output_shapes	
:*
dtype0

 Nadam/Playlists_Dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Nadam/Playlists_Dense_2/kernel/m

4Nadam/Playlists_Dense_2/kernel/m/Read/ReadVariableOpReadVariableOp Nadam/Playlists_Dense_2/kernel/m* 
_output_shapes
:
*
dtype0

Nadam/Playlists_Dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Nadam/Playlists_Dense_2/bias/m

2Nadam/Playlists_Dense_2/bias/m/Read/ReadVariableOpReadVariableOpNadam/Playlists_Dense_2/bias/m*
_output_shapes	
:*
dtype0

 Nadam/Playlists_Dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Nadam/Playlists_Dense_3/kernel/m

4Nadam/Playlists_Dense_3/kernel/m/Read/ReadVariableOpReadVariableOp Nadam/Playlists_Dense_3/kernel/m* 
_output_shapes
:
*
dtype0

Nadam/Playlists_Dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Nadam/Playlists_Dense_3/bias/m

2Nadam/Playlists_Dense_3/bias/m/Read/ReadVariableOpReadVariableOpNadam/Playlists_Dense_3/bias/m*
_output_shapes	
:*
dtype0

Nadam/Tracks_Dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*.
shared_nameNadam/Tracks_Dense_1/kernel/m

1Nadam/Tracks_Dense_1/kernel/m/Read/ReadVariableOpReadVariableOpNadam/Tracks_Dense_1/kernel/m*
_output_shapes
:	*
dtype0

Nadam/Tracks_Dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameNadam/Tracks_Dense_1/bias/m

/Nadam/Tracks_Dense_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/Tracks_Dense_1/bias/m*
_output_shapes	
:*
dtype0

Nadam/Playlists_Latent/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*0
shared_name!Nadam/Playlists_Latent/kernel/m

3Nadam/Playlists_Latent/kernel/m/Read/ReadVariableOpReadVariableOpNadam/Playlists_Latent/kernel/m*
_output_shapes
:	@*
dtype0

Nadam/Playlists_Latent/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameNadam/Playlists_Latent/bias/m

1Nadam/Playlists_Latent/bias/m/Read/ReadVariableOpReadVariableOpNadam/Playlists_Latent/bias/m*
_output_shapes
:@*
dtype0

Nadam/Tracks_Latent/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*-
shared_nameNadam/Tracks_Latent/kernel/m

0Nadam/Tracks_Latent/kernel/m/Read/ReadVariableOpReadVariableOpNadam/Tracks_Latent/kernel/m*
_output_shapes
:	@*
dtype0

Nadam/Tracks_Latent/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameNadam/Tracks_Latent/bias/m

.Nadam/Tracks_Latent/bias/m/Read/ReadVariableOpReadVariableOpNadam/Tracks_Latent/bias/m*
_output_shapes
:@*
dtype0

Nadam/Prediction_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameNadam/Prediction_1/kernel/m

/Nadam/Prediction_1/kernel/m/Read/ReadVariableOpReadVariableOpNadam/Prediction_1/kernel/m* 
_output_shapes
:
*
dtype0

Nadam/Prediction_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameNadam/Prediction_1/bias/m

-Nadam/Prediction_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/Prediction_1/bias/m*
_output_shapes	
:*
dtype0

Nadam/Prediction_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameNadam/Prediction_2/kernel/m

/Nadam/Prediction_2/kernel/m/Read/ReadVariableOpReadVariableOpNadam/Prediction_2/kernel/m* 
_output_shapes
:
*
dtype0

Nadam/Prediction_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameNadam/Prediction_2/bias/m

-Nadam/Prediction_2/bias/m/Read/ReadVariableOpReadVariableOpNadam/Prediction_2/bias/m*
_output_shapes	
:*
dtype0

Nadam/Prediction_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameNadam/Prediction_3/kernel/m

/Nadam/Prediction_3/kernel/m/Read/ReadVariableOpReadVariableOpNadam/Prediction_3/kernel/m* 
_output_shapes
:
*
dtype0

Nadam/Prediction_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameNadam/Prediction_3/bias/m

-Nadam/Prediction_3/bias/m/Read/ReadVariableOpReadVariableOpNadam/Prediction_3/bias/m*
_output_shapes	
:*
dtype0

Nadam/Prediction_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameNadam/Prediction_4/kernel/m

/Nadam/Prediction_4/kernel/m/Read/ReadVariableOpReadVariableOpNadam/Prediction_4/kernel/m* 
_output_shapes
:
*
dtype0

Nadam/Prediction_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameNadam/Prediction_4/bias/m

-Nadam/Prediction_4/bias/m/Read/ReadVariableOpReadVariableOpNadam/Prediction_4/bias/m*
_output_shapes	
:*
dtype0

Nadam/Final_Prediction/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*0
shared_name!Nadam/Final_Prediction/kernel/m

3Nadam/Final_Prediction/kernel/m/Read/ReadVariableOpReadVariableOpNadam/Final_Prediction/kernel/m*
_output_shapes
:	*
dtype0

Nadam/Final_Prediction/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameNadam/Final_Prediction/bias/m

1Nadam/Final_Prediction/bias/m/Read/ReadVariableOpReadVariableOpNadam/Final_Prediction/bias/m*
_output_shapes
:*
dtype0

 Nadam/Playlists_Dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Nadam/Playlists_Dense_1/kernel/v

4Nadam/Playlists_Dense_1/kernel/v/Read/ReadVariableOpReadVariableOp Nadam/Playlists_Dense_1/kernel/v* 
_output_shapes
:
*
dtype0

Nadam/Playlists_Dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Nadam/Playlists_Dense_1/bias/v

2Nadam/Playlists_Dense_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/Playlists_Dense_1/bias/v*
_output_shapes	
:*
dtype0

 Nadam/Playlists_Dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Nadam/Playlists_Dense_2/kernel/v

4Nadam/Playlists_Dense_2/kernel/v/Read/ReadVariableOpReadVariableOp Nadam/Playlists_Dense_2/kernel/v* 
_output_shapes
:
*
dtype0

Nadam/Playlists_Dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Nadam/Playlists_Dense_2/bias/v

2Nadam/Playlists_Dense_2/bias/v/Read/ReadVariableOpReadVariableOpNadam/Playlists_Dense_2/bias/v*
_output_shapes	
:*
dtype0

 Nadam/Playlists_Dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Nadam/Playlists_Dense_3/kernel/v

4Nadam/Playlists_Dense_3/kernel/v/Read/ReadVariableOpReadVariableOp Nadam/Playlists_Dense_3/kernel/v* 
_output_shapes
:
*
dtype0

Nadam/Playlists_Dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Nadam/Playlists_Dense_3/bias/v

2Nadam/Playlists_Dense_3/bias/v/Read/ReadVariableOpReadVariableOpNadam/Playlists_Dense_3/bias/v*
_output_shapes	
:*
dtype0

Nadam/Tracks_Dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*.
shared_nameNadam/Tracks_Dense_1/kernel/v

1Nadam/Tracks_Dense_1/kernel/v/Read/ReadVariableOpReadVariableOpNadam/Tracks_Dense_1/kernel/v*
_output_shapes
:	*
dtype0

Nadam/Tracks_Dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameNadam/Tracks_Dense_1/bias/v

/Nadam/Tracks_Dense_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/Tracks_Dense_1/bias/v*
_output_shapes	
:*
dtype0

Nadam/Playlists_Latent/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*0
shared_name!Nadam/Playlists_Latent/kernel/v

3Nadam/Playlists_Latent/kernel/v/Read/ReadVariableOpReadVariableOpNadam/Playlists_Latent/kernel/v*
_output_shapes
:	@*
dtype0

Nadam/Playlists_Latent/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameNadam/Playlists_Latent/bias/v

1Nadam/Playlists_Latent/bias/v/Read/ReadVariableOpReadVariableOpNadam/Playlists_Latent/bias/v*
_output_shapes
:@*
dtype0

Nadam/Tracks_Latent/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*-
shared_nameNadam/Tracks_Latent/kernel/v

0Nadam/Tracks_Latent/kernel/v/Read/ReadVariableOpReadVariableOpNadam/Tracks_Latent/kernel/v*
_output_shapes
:	@*
dtype0

Nadam/Tracks_Latent/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameNadam/Tracks_Latent/bias/v

.Nadam/Tracks_Latent/bias/v/Read/ReadVariableOpReadVariableOpNadam/Tracks_Latent/bias/v*
_output_shapes
:@*
dtype0

Nadam/Prediction_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameNadam/Prediction_1/kernel/v

/Nadam/Prediction_1/kernel/v/Read/ReadVariableOpReadVariableOpNadam/Prediction_1/kernel/v* 
_output_shapes
:
*
dtype0

Nadam/Prediction_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameNadam/Prediction_1/bias/v

-Nadam/Prediction_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/Prediction_1/bias/v*
_output_shapes	
:*
dtype0

Nadam/Prediction_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameNadam/Prediction_2/kernel/v

/Nadam/Prediction_2/kernel/v/Read/ReadVariableOpReadVariableOpNadam/Prediction_2/kernel/v* 
_output_shapes
:
*
dtype0

Nadam/Prediction_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameNadam/Prediction_2/bias/v

-Nadam/Prediction_2/bias/v/Read/ReadVariableOpReadVariableOpNadam/Prediction_2/bias/v*
_output_shapes	
:*
dtype0

Nadam/Prediction_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameNadam/Prediction_3/kernel/v

/Nadam/Prediction_3/kernel/v/Read/ReadVariableOpReadVariableOpNadam/Prediction_3/kernel/v* 
_output_shapes
:
*
dtype0

Nadam/Prediction_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameNadam/Prediction_3/bias/v

-Nadam/Prediction_3/bias/v/Read/ReadVariableOpReadVariableOpNadam/Prediction_3/bias/v*
_output_shapes	
:*
dtype0

Nadam/Prediction_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameNadam/Prediction_4/kernel/v

/Nadam/Prediction_4/kernel/v/Read/ReadVariableOpReadVariableOpNadam/Prediction_4/kernel/v* 
_output_shapes
:
*
dtype0

Nadam/Prediction_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameNadam/Prediction_4/bias/v

-Nadam/Prediction_4/bias/v/Read/ReadVariableOpReadVariableOpNadam/Prediction_4/bias/v*
_output_shapes	
:*
dtype0

Nadam/Final_Prediction/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*0
shared_name!Nadam/Final_Prediction/kernel/v

3Nadam/Final_Prediction/kernel/v/Read/ReadVariableOpReadVariableOpNadam/Final_Prediction/kernel/v*
_output_shapes
:	*
dtype0

Nadam/Final_Prediction/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameNadam/Final_Prediction/bias/v

1Nadam/Final_Prediction/bias/v/Read/ReadVariableOpReadVariableOpNadam/Final_Prediction/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ø
valueÍBÉ BÁ
³
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
¦

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
* 
¦

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses*
¦

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*
¦

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses*
¦

Gkernel
Hbias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses*

O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses* 
¦

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses*
¦

]kernel
^bias
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses*
¦

ekernel
fbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses*
¦

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses*
¦

ukernel
vbias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses*

}iter

~beta_1

beta_2

decay
learning_rate
momentum_cachemÕ mÖ'm×(mØ/mÙ0mÚ7mÛ8mÜ?mÝ@mÞGmßHmàUmáVmâ]mã^mäemåfmæmmçnmèumévmêvë vì'ví(vî/vï0vð7vñ8vò?vó@vôGvõHvöUv÷Vvø]vù^vúevûfvümvýnvþuvÿvv*
ª
0
 1
'2
(3
/4
05
76
87
?8
@9
G10
H11
U12
V13
]14
^15
e16
f17
m18
n19
u20
v21*
ª
0
 1
'2
(3
/4
05
76
87
?8
@9
G10
H11
U12
V13
]14
^15
e16
f17
m18
n19
u20
v21*
* 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

serving_default* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
hb
VARIABLE_VALUEPlaylists_Dense_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEPlaylists_Dense_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
hb
VARIABLE_VALUEPlaylists_Dense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEPlaylists_Dense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
hb
VARIABLE_VALUEPlaylists_Dense_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEPlaylists_Dense_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

/0
01*

/0
01*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUETracks_Dense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUETracks_Dense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

70
81*

70
81*
* 

non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
ga
VARIABLE_VALUEPlaylists_Latent/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEPlaylists_Latent/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
@1*

?0
@1*
* 

¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*
* 
* 
d^
VARIABLE_VALUETracks_Latent/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUETracks_Latent/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

G0
H1*

G0
H1*
* 

§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses* 
* 
* 
c]
VARIABLE_VALUEPrediction_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEPrediction_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

U0
V1*

U0
V1*
* 

±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEPrediction_2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEPrediction_2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

]0
^1*

]0
^1*
* 

¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEPrediction_3/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEPrediction_3/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

e0
f1*

e0
f1*
* 

»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEPrediction_4/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEPrediction_4/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

m0
n1*

m0
n1*
* 

Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*
* 
* 
hb
VARIABLE_VALUEFinal_Prediction/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEFinal_Prediction/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*

u0
v1*

u0
v1*
* 

Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*
* 
* 
MG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE*
* 
r
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14*

Ê0
Ë1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

Ìtotal

Ícount
Î	variables
Ï	keras_api*
M

Ðtotal

Ñcount
Ò
_fn_kwargs
Ó	variables
Ô	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ì0
Í1*

Î	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ð0
Ñ1*

Ó	variables*

VARIABLE_VALUE Nadam/Playlists_Dense_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Playlists_Dense_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Nadam/Playlists_Dense_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Playlists_Dense_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Nadam/Playlists_Dense_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Playlists_Dense_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Tracks_Dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Tracks_Dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Playlists_Latent/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Playlists_Latent/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Tracks_Latent/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUENadam/Tracks_Latent/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Prediction_1/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUENadam/Prediction_1/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Prediction_2/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUENadam/Prediction_2/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Prediction_3/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUENadam/Prediction_3/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Prediction_4/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUENadam/Prediction_4/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Final_Prediction/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Final_Prediction/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Nadam/Playlists_Dense_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Playlists_Dense_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Nadam/Playlists_Dense_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Playlists_Dense_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Nadam/Playlists_Dense_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Playlists_Dense_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Tracks_Dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Tracks_Dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Playlists_Latent/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Playlists_Latent/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Tracks_Latent/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUENadam/Tracks_Latent/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Prediction_1/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUENadam/Prediction_1/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Prediction_2/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUENadam/Prediction_2/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Prediction_3/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUENadam/Prediction_3/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Prediction_4/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUENadam/Prediction_4/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Final_Prediction/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUENadam/Final_Prediction/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

$serving_default_Audio_Feature_InputsPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

serving_default_Playlist_InputsPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ

ø
StatefulPartitionedCallStatefulPartitionedCall$serving_default_Audio_Feature_Inputsserving_default_Playlist_InputsPlaylists_Dense_1/kernelPlaylists_Dense_1/biasPlaylists_Dense_2/kernelPlaylists_Dense_2/biasTracks_Dense_1/kernelTracks_Dense_1/biasPlaylists_Dense_3/kernelPlaylists_Dense_3/biasPlaylists_Latent/kernelPlaylists_Latent/biasTracks_Latent/kernelTracks_Latent/biasPrediction_1/kernelPrediction_1/biasPrediction_2/kernelPrediction_2/biasPrediction_3/kernelPrediction_3/biasPrediction_4/kernelPrediction_4/biasFinal_Prediction/kernelFinal_Prediction/bias*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_41309
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ú
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,Playlists_Dense_1/kernel/Read/ReadVariableOp*Playlists_Dense_1/bias/Read/ReadVariableOp,Playlists_Dense_2/kernel/Read/ReadVariableOp*Playlists_Dense_2/bias/Read/ReadVariableOp,Playlists_Dense_3/kernel/Read/ReadVariableOp*Playlists_Dense_3/bias/Read/ReadVariableOp)Tracks_Dense_1/kernel/Read/ReadVariableOp'Tracks_Dense_1/bias/Read/ReadVariableOp+Playlists_Latent/kernel/Read/ReadVariableOp)Playlists_Latent/bias/Read/ReadVariableOp(Tracks_Latent/kernel/Read/ReadVariableOp&Tracks_Latent/bias/Read/ReadVariableOp'Prediction_1/kernel/Read/ReadVariableOp%Prediction_1/bias/Read/ReadVariableOp'Prediction_2/kernel/Read/ReadVariableOp%Prediction_2/bias/Read/ReadVariableOp'Prediction_3/kernel/Read/ReadVariableOp%Prediction_3/bias/Read/ReadVariableOp'Prediction_4/kernel/Read/ReadVariableOp%Prediction_4/bias/Read/ReadVariableOp+Final_Prediction/kernel/Read/ReadVariableOp)Final_Prediction/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp4Nadam/Playlists_Dense_1/kernel/m/Read/ReadVariableOp2Nadam/Playlists_Dense_1/bias/m/Read/ReadVariableOp4Nadam/Playlists_Dense_2/kernel/m/Read/ReadVariableOp2Nadam/Playlists_Dense_2/bias/m/Read/ReadVariableOp4Nadam/Playlists_Dense_3/kernel/m/Read/ReadVariableOp2Nadam/Playlists_Dense_3/bias/m/Read/ReadVariableOp1Nadam/Tracks_Dense_1/kernel/m/Read/ReadVariableOp/Nadam/Tracks_Dense_1/bias/m/Read/ReadVariableOp3Nadam/Playlists_Latent/kernel/m/Read/ReadVariableOp1Nadam/Playlists_Latent/bias/m/Read/ReadVariableOp0Nadam/Tracks_Latent/kernel/m/Read/ReadVariableOp.Nadam/Tracks_Latent/bias/m/Read/ReadVariableOp/Nadam/Prediction_1/kernel/m/Read/ReadVariableOp-Nadam/Prediction_1/bias/m/Read/ReadVariableOp/Nadam/Prediction_2/kernel/m/Read/ReadVariableOp-Nadam/Prediction_2/bias/m/Read/ReadVariableOp/Nadam/Prediction_3/kernel/m/Read/ReadVariableOp-Nadam/Prediction_3/bias/m/Read/ReadVariableOp/Nadam/Prediction_4/kernel/m/Read/ReadVariableOp-Nadam/Prediction_4/bias/m/Read/ReadVariableOp3Nadam/Final_Prediction/kernel/m/Read/ReadVariableOp1Nadam/Final_Prediction/bias/m/Read/ReadVariableOp4Nadam/Playlists_Dense_1/kernel/v/Read/ReadVariableOp2Nadam/Playlists_Dense_1/bias/v/Read/ReadVariableOp4Nadam/Playlists_Dense_2/kernel/v/Read/ReadVariableOp2Nadam/Playlists_Dense_2/bias/v/Read/ReadVariableOp4Nadam/Playlists_Dense_3/kernel/v/Read/ReadVariableOp2Nadam/Playlists_Dense_3/bias/v/Read/ReadVariableOp1Nadam/Tracks_Dense_1/kernel/v/Read/ReadVariableOp/Nadam/Tracks_Dense_1/bias/v/Read/ReadVariableOp3Nadam/Playlists_Latent/kernel/v/Read/ReadVariableOp1Nadam/Playlists_Latent/bias/v/Read/ReadVariableOp0Nadam/Tracks_Latent/kernel/v/Read/ReadVariableOp.Nadam/Tracks_Latent/bias/v/Read/ReadVariableOp/Nadam/Prediction_1/kernel/v/Read/ReadVariableOp-Nadam/Prediction_1/bias/v/Read/ReadVariableOp/Nadam/Prediction_2/kernel/v/Read/ReadVariableOp-Nadam/Prediction_2/bias/v/Read/ReadVariableOp/Nadam/Prediction_3/kernel/v/Read/ReadVariableOp-Nadam/Prediction_3/bias/v/Read/ReadVariableOp/Nadam/Prediction_4/kernel/v/Read/ReadVariableOp-Nadam/Prediction_4/bias/v/Read/ReadVariableOp3Nadam/Final_Prediction/kernel/v/Read/ReadVariableOp1Nadam/Final_Prediction/bias/v/Read/ReadVariableOpConst*Y
TinR
P2N	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_41805

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamePlaylists_Dense_1/kernelPlaylists_Dense_1/biasPlaylists_Dense_2/kernelPlaylists_Dense_2/biasPlaylists_Dense_3/kernelPlaylists_Dense_3/biasTracks_Dense_1/kernelTracks_Dense_1/biasPlaylists_Latent/kernelPlaylists_Latent/biasTracks_Latent/kernelTracks_Latent/biasPrediction_1/kernelPrediction_1/biasPrediction_2/kernelPrediction_2/biasPrediction_3/kernelPrediction_3/biasPrediction_4/kernelPrediction_4/biasFinal_Prediction/kernelFinal_Prediction/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcounttotal_1count_1 Nadam/Playlists_Dense_1/kernel/mNadam/Playlists_Dense_1/bias/m Nadam/Playlists_Dense_2/kernel/mNadam/Playlists_Dense_2/bias/m Nadam/Playlists_Dense_3/kernel/mNadam/Playlists_Dense_3/bias/mNadam/Tracks_Dense_1/kernel/mNadam/Tracks_Dense_1/bias/mNadam/Playlists_Latent/kernel/mNadam/Playlists_Latent/bias/mNadam/Tracks_Latent/kernel/mNadam/Tracks_Latent/bias/mNadam/Prediction_1/kernel/mNadam/Prediction_1/bias/mNadam/Prediction_2/kernel/mNadam/Prediction_2/bias/mNadam/Prediction_3/kernel/mNadam/Prediction_3/bias/mNadam/Prediction_4/kernel/mNadam/Prediction_4/bias/mNadam/Final_Prediction/kernel/mNadam/Final_Prediction/bias/m Nadam/Playlists_Dense_1/kernel/vNadam/Playlists_Dense_1/bias/v Nadam/Playlists_Dense_2/kernel/vNadam/Playlists_Dense_2/bias/v Nadam/Playlists_Dense_3/kernel/vNadam/Playlists_Dense_3/bias/vNadam/Tracks_Dense_1/kernel/vNadam/Tracks_Dense_1/bias/vNadam/Playlists_Latent/kernel/vNadam/Playlists_Latent/bias/vNadam/Tracks_Latent/kernel/vNadam/Tracks_Latent/bias/vNadam/Prediction_1/kernel/vNadam/Prediction_1/bias/vNadam/Prediction_2/kernel/vNadam/Prediction_2/bias/vNadam/Prediction_3/kernel/vNadam/Prediction_3/bias/vNadam/Prediction_4/kernel/vNadam/Prediction_4/bias/vNadam/Final_Prediction/kernel/vNadam/Final_Prediction/bias/v*X
TinQ
O2M*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_42043´ÿ
ª

û
G__inference_Prediction_3_layer_call_and_return_conditional_losses_41513

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª

û
G__inference_Prediction_4_layer_call_and_return_conditional_losses_40449

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
¡
1__inference_Playlists_Dense_3_layer_call_fn_41369

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Playlists_Dense_3_layer_call_and_return_conditional_losses_40338p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð

.__inference_Tracks_Dense_1_layer_call_fn_41389

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Tracks_Dense_1_layer_call_and_return_conditional_losses_40321p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

,__inference_Prediction_2_layer_call_fn_41482

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_2_layer_call_and_return_conditional_losses_40415p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

ü
I__inference_Tracks_Dense_1_layer_call_and_return_conditional_losses_40321

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


1__inference_Track_Recommender_layer_call_fn_40855
playlist_inputs
audio_feature_inputs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	@
	unknown_8:@
	unknown_9:	@

unknown_10:@

unknown_11:


unknown_12:	

unknown_13:


unknown_14:	

unknown_15:


unknown_16:	

unknown_17:


unknown_18:	

unknown_19:	

unknown_20:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallplaylist_inputsaudio_feature_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_40758o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)
_user_specified_namePlaylist_Inputs:]Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_nameAudio_Feature_Inputs
¥

ý
K__inference_Final_Prediction_layer_call_and_return_conditional_losses_41553

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª

û
G__inference_Prediction_2_layer_call_and_return_conditional_losses_41493

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
¡
1__inference_Playlists_Dense_1_layer_call_fn_41329

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Playlists_Dense_1_layer_call_and_return_conditional_losses_40287p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
Ã"
__inference__traced_save_41805
file_prefix7
3savev2_playlists_dense_1_kernel_read_readvariableop5
1savev2_playlists_dense_1_bias_read_readvariableop7
3savev2_playlists_dense_2_kernel_read_readvariableop5
1savev2_playlists_dense_2_bias_read_readvariableop7
3savev2_playlists_dense_3_kernel_read_readvariableop5
1savev2_playlists_dense_3_bias_read_readvariableop4
0savev2_tracks_dense_1_kernel_read_readvariableop2
.savev2_tracks_dense_1_bias_read_readvariableop6
2savev2_playlists_latent_kernel_read_readvariableop4
0savev2_playlists_latent_bias_read_readvariableop3
/savev2_tracks_latent_kernel_read_readvariableop1
-savev2_tracks_latent_bias_read_readvariableop2
.savev2_prediction_1_kernel_read_readvariableop0
,savev2_prediction_1_bias_read_readvariableop2
.savev2_prediction_2_kernel_read_readvariableop0
,savev2_prediction_2_bias_read_readvariableop2
.savev2_prediction_3_kernel_read_readvariableop0
,savev2_prediction_3_bias_read_readvariableop2
.savev2_prediction_4_kernel_read_readvariableop0
,savev2_prediction_4_bias_read_readvariableop6
2savev2_final_prediction_kernel_read_readvariableop4
0savev2_final_prediction_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop?
;savev2_nadam_playlists_dense_1_kernel_m_read_readvariableop=
9savev2_nadam_playlists_dense_1_bias_m_read_readvariableop?
;savev2_nadam_playlists_dense_2_kernel_m_read_readvariableop=
9savev2_nadam_playlists_dense_2_bias_m_read_readvariableop?
;savev2_nadam_playlists_dense_3_kernel_m_read_readvariableop=
9savev2_nadam_playlists_dense_3_bias_m_read_readvariableop<
8savev2_nadam_tracks_dense_1_kernel_m_read_readvariableop:
6savev2_nadam_tracks_dense_1_bias_m_read_readvariableop>
:savev2_nadam_playlists_latent_kernel_m_read_readvariableop<
8savev2_nadam_playlists_latent_bias_m_read_readvariableop;
7savev2_nadam_tracks_latent_kernel_m_read_readvariableop9
5savev2_nadam_tracks_latent_bias_m_read_readvariableop:
6savev2_nadam_prediction_1_kernel_m_read_readvariableop8
4savev2_nadam_prediction_1_bias_m_read_readvariableop:
6savev2_nadam_prediction_2_kernel_m_read_readvariableop8
4savev2_nadam_prediction_2_bias_m_read_readvariableop:
6savev2_nadam_prediction_3_kernel_m_read_readvariableop8
4savev2_nadam_prediction_3_bias_m_read_readvariableop:
6savev2_nadam_prediction_4_kernel_m_read_readvariableop8
4savev2_nadam_prediction_4_bias_m_read_readvariableop>
:savev2_nadam_final_prediction_kernel_m_read_readvariableop<
8savev2_nadam_final_prediction_bias_m_read_readvariableop?
;savev2_nadam_playlists_dense_1_kernel_v_read_readvariableop=
9savev2_nadam_playlists_dense_1_bias_v_read_readvariableop?
;savev2_nadam_playlists_dense_2_kernel_v_read_readvariableop=
9savev2_nadam_playlists_dense_2_bias_v_read_readvariableop?
;savev2_nadam_playlists_dense_3_kernel_v_read_readvariableop=
9savev2_nadam_playlists_dense_3_bias_v_read_readvariableop<
8savev2_nadam_tracks_dense_1_kernel_v_read_readvariableop:
6savev2_nadam_tracks_dense_1_bias_v_read_readvariableop>
:savev2_nadam_playlists_latent_kernel_v_read_readvariableop<
8savev2_nadam_playlists_latent_bias_v_read_readvariableop;
7savev2_nadam_tracks_latent_kernel_v_read_readvariableop9
5savev2_nadam_tracks_latent_bias_v_read_readvariableop:
6savev2_nadam_prediction_1_kernel_v_read_readvariableop8
4savev2_nadam_prediction_1_bias_v_read_readvariableop:
6savev2_nadam_prediction_2_kernel_v_read_readvariableop8
4savev2_nadam_prediction_2_bias_v_read_readvariableop:
6savev2_nadam_prediction_3_kernel_v_read_readvariableop8
4savev2_nadam_prediction_3_bias_v_read_readvariableop:
6savev2_nadam_prediction_4_kernel_v_read_readvariableop8
4savev2_nadam_prediction_4_bias_v_read_readvariableop>
:savev2_nadam_final_prediction_kernel_v_read_readvariableop<
8savev2_nadam_final_prediction_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: +
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*©*
value*B*MB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*¯
value¥B¢MB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¤!
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_playlists_dense_1_kernel_read_readvariableop1savev2_playlists_dense_1_bias_read_readvariableop3savev2_playlists_dense_2_kernel_read_readvariableop1savev2_playlists_dense_2_bias_read_readvariableop3savev2_playlists_dense_3_kernel_read_readvariableop1savev2_playlists_dense_3_bias_read_readvariableop0savev2_tracks_dense_1_kernel_read_readvariableop.savev2_tracks_dense_1_bias_read_readvariableop2savev2_playlists_latent_kernel_read_readvariableop0savev2_playlists_latent_bias_read_readvariableop/savev2_tracks_latent_kernel_read_readvariableop-savev2_tracks_latent_bias_read_readvariableop.savev2_prediction_1_kernel_read_readvariableop,savev2_prediction_1_bias_read_readvariableop.savev2_prediction_2_kernel_read_readvariableop,savev2_prediction_2_bias_read_readvariableop.savev2_prediction_3_kernel_read_readvariableop,savev2_prediction_3_bias_read_readvariableop.savev2_prediction_4_kernel_read_readvariableop,savev2_prediction_4_bias_read_readvariableop2savev2_final_prediction_kernel_read_readvariableop0savev2_final_prediction_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop;savev2_nadam_playlists_dense_1_kernel_m_read_readvariableop9savev2_nadam_playlists_dense_1_bias_m_read_readvariableop;savev2_nadam_playlists_dense_2_kernel_m_read_readvariableop9savev2_nadam_playlists_dense_2_bias_m_read_readvariableop;savev2_nadam_playlists_dense_3_kernel_m_read_readvariableop9savev2_nadam_playlists_dense_3_bias_m_read_readvariableop8savev2_nadam_tracks_dense_1_kernel_m_read_readvariableop6savev2_nadam_tracks_dense_1_bias_m_read_readvariableop:savev2_nadam_playlists_latent_kernel_m_read_readvariableop8savev2_nadam_playlists_latent_bias_m_read_readvariableop7savev2_nadam_tracks_latent_kernel_m_read_readvariableop5savev2_nadam_tracks_latent_bias_m_read_readvariableop6savev2_nadam_prediction_1_kernel_m_read_readvariableop4savev2_nadam_prediction_1_bias_m_read_readvariableop6savev2_nadam_prediction_2_kernel_m_read_readvariableop4savev2_nadam_prediction_2_bias_m_read_readvariableop6savev2_nadam_prediction_3_kernel_m_read_readvariableop4savev2_nadam_prediction_3_bias_m_read_readvariableop6savev2_nadam_prediction_4_kernel_m_read_readvariableop4savev2_nadam_prediction_4_bias_m_read_readvariableop:savev2_nadam_final_prediction_kernel_m_read_readvariableop8savev2_nadam_final_prediction_bias_m_read_readvariableop;savev2_nadam_playlists_dense_1_kernel_v_read_readvariableop9savev2_nadam_playlists_dense_1_bias_v_read_readvariableop;savev2_nadam_playlists_dense_2_kernel_v_read_readvariableop9savev2_nadam_playlists_dense_2_bias_v_read_readvariableop;savev2_nadam_playlists_dense_3_kernel_v_read_readvariableop9savev2_nadam_playlists_dense_3_bias_v_read_readvariableop8savev2_nadam_tracks_dense_1_kernel_v_read_readvariableop6savev2_nadam_tracks_dense_1_bias_v_read_readvariableop:savev2_nadam_playlists_latent_kernel_v_read_readvariableop8savev2_nadam_playlists_latent_bias_v_read_readvariableop7savev2_nadam_tracks_latent_kernel_v_read_readvariableop5savev2_nadam_tracks_latent_bias_v_read_readvariableop6savev2_nadam_prediction_1_kernel_v_read_readvariableop4savev2_nadam_prediction_1_bias_v_read_readvariableop6savev2_nadam_prediction_2_kernel_v_read_readvariableop4savev2_nadam_prediction_2_bias_v_read_readvariableop6savev2_nadam_prediction_3_kernel_v_read_readvariableop4savev2_nadam_prediction_3_bias_v_read_readvariableop6savev2_nadam_prediction_4_kernel_v_read_readvariableop4savev2_nadam_prediction_4_bias_v_read_readvariableop:savev2_nadam_final_prediction_kernel_v_read_readvariableop8savev2_nadam_final_prediction_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *[
dtypesQ
O2M	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesù
ö: :
::
::
::	::	@:@:	@:@:
::
::
::
::	:: : : : : : : : : : :
::
::
::	::	@:@:	@:@:
::
::
::
::	::
::
::
::	::	@:@:	@:@:
::
::
::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	:!

_output_shapes	
::%	!

_output_shapes
:	@: 


_output_shapes
:@:%!

_output_shapes
:	@: 

_output_shapes
:@:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :&!"
 
_output_shapes
:
:!"

_output_shapes	
::&#"
 
_output_shapes
:
:!$

_output_shapes	
::&%"
 
_output_shapes
:
:!&

_output_shapes	
::%'!

_output_shapes
:	:!(

_output_shapes	
::%)!

_output_shapes
:	@: *

_output_shapes
:@:%+!

_output_shapes
:	@: ,

_output_shapes
:@:&-"
 
_output_shapes
:
:!.

_output_shapes	
::&/"
 
_output_shapes
:
:!0

_output_shapes	
::&1"
 
_output_shapes
:
:!2

_output_shapes	
::&3"
 
_output_shapes
:
:!4

_output_shapes	
::%5!

_output_shapes
:	: 6

_output_shapes
::&7"
 
_output_shapes
:
:!8

_output_shapes	
::&9"
 
_output_shapes
:
:!:

_output_shapes	
::&;"
 
_output_shapes
:
:!<

_output_shapes	
::%=!

_output_shapes
:	:!>

_output_shapes	
::%?!

_output_shapes
:	@: @

_output_shapes
:@:%A!

_output_shapes
:	@: B

_output_shapes
:@:&C"
 
_output_shapes
:
:!D

_output_shapes	
::&E"
 
_output_shapes
:
:!F

_output_shapes	
::&G"
 
_output_shapes
:
:!H

_output_shapes	
::&I"
 
_output_shapes
:
:!J

_output_shapes	
::%K!

_output_shapes
:	: L

_output_shapes
::M

_output_shapes
: 
ª

û
G__inference_Prediction_1_layer_call_and_return_conditional_losses_40398

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²E
¦
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_40758

inputs
inputs_1+
playlists_dense_1_40701:
&
playlists_dense_1_40703:	+
playlists_dense_2_40706:
&
playlists_dense_2_40708:	'
tracks_dense_1_40711:	#
tracks_dense_1_40713:	+
playlists_dense_3_40716:
&
playlists_dense_3_40718:	)
playlists_latent_40721:	@$
playlists_latent_40723:@&
tracks_latent_40726:	@!
tracks_latent_40728:@&
prediction_1_40732:
!
prediction_1_40734:	&
prediction_2_40737:
!
prediction_2_40739:	&
prediction_3_40742:
!
prediction_3_40744:	&
prediction_4_40747:
!
prediction_4_40749:	)
final_prediction_40752:	$
final_prediction_40754:
identity¢(Final_Prediction/StatefulPartitionedCall¢)Playlists_Dense_1/StatefulPartitionedCall¢)Playlists_Dense_2/StatefulPartitionedCall¢)Playlists_Dense_3/StatefulPartitionedCall¢(Playlists_Latent/StatefulPartitionedCall¢$Prediction_1/StatefulPartitionedCall¢$Prediction_2/StatefulPartitionedCall¢$Prediction_3/StatefulPartitionedCall¢$Prediction_4/StatefulPartitionedCall¢&Tracks_Dense_1/StatefulPartitionedCall¢%Tracks_Latent/StatefulPartitionedCallÎ
#Platlists_Flattened/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Platlists_Flattened_layer_call_and_return_conditional_losses_40274¸
)Playlists_Dense_1/StatefulPartitionedCallStatefulPartitionedCall,Platlists_Flattened/PartitionedCall:output:0playlists_dense_1_40701playlists_dense_1_40703*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Playlists_Dense_1_layer_call_and_return_conditional_losses_40287¾
)Playlists_Dense_2/StatefulPartitionedCallStatefulPartitionedCall2Playlists_Dense_1/StatefulPartitionedCall:output:0playlists_dense_2_40706playlists_dense_2_40708*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Playlists_Dense_2_layer_call_and_return_conditional_losses_40304
&Tracks_Dense_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1tracks_dense_1_40711tracks_dense_1_40713*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Tracks_Dense_1_layer_call_and_return_conditional_losses_40321¾
)Playlists_Dense_3/StatefulPartitionedCallStatefulPartitionedCall2Playlists_Dense_2/StatefulPartitionedCall:output:0playlists_dense_3_40716playlists_dense_3_40718*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Playlists_Dense_3_layer_call_and_return_conditional_losses_40338¹
(Playlists_Latent/StatefulPartitionedCallStatefulPartitionedCall2Playlists_Dense_3/StatefulPartitionedCall:output:0playlists_latent_40721playlists_latent_40723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Playlists_Latent_layer_call_and_return_conditional_losses_40355ª
%Tracks_Latent/StatefulPartitionedCallStatefulPartitionedCall/Tracks_Dense_1/StatefulPartitionedCall:output:0tracks_latent_40726tracks_latent_40728*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Tracks_Latent_layer_call_and_return_conditional_losses_40372°
&Concatenated_Encodings/PartitionedCallPartitionedCall1Playlists_Latent/StatefulPartitionedCall:output:0.Tracks_Latent/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Concatenated_Encodings_layer_call_and_return_conditional_losses_40385§
$Prediction_1/StatefulPartitionedCallStatefulPartitionedCall/Concatenated_Encodings/PartitionedCall:output:0prediction_1_40732prediction_1_40734*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_1_layer_call_and_return_conditional_losses_40398¥
$Prediction_2/StatefulPartitionedCallStatefulPartitionedCall-Prediction_1/StatefulPartitionedCall:output:0prediction_2_40737prediction_2_40739*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_2_layer_call_and_return_conditional_losses_40415¥
$Prediction_3/StatefulPartitionedCallStatefulPartitionedCall-Prediction_2/StatefulPartitionedCall:output:0prediction_3_40742prediction_3_40744*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_3_layer_call_and_return_conditional_losses_40432¥
$Prediction_4/StatefulPartitionedCallStatefulPartitionedCall-Prediction_3/StatefulPartitionedCall:output:0prediction_4_40747prediction_4_40749*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_4_layer_call_and_return_conditional_losses_40449´
(Final_Prediction/StatefulPartitionedCallStatefulPartitionedCall-Prediction_4/StatefulPartitionedCall:output:0final_prediction_40752final_prediction_40754*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Final_Prediction_layer_call_and_return_conditional_losses_40466
IdentityIdentity1Final_Prediction/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^Final_Prediction/StatefulPartitionedCall*^Playlists_Dense_1/StatefulPartitionedCall*^Playlists_Dense_2/StatefulPartitionedCall*^Playlists_Dense_3/StatefulPartitionedCall)^Playlists_Latent/StatefulPartitionedCall%^Prediction_1/StatefulPartitionedCall%^Prediction_2/StatefulPartitionedCall%^Prediction_3/StatefulPartitionedCall%^Prediction_4/StatefulPartitionedCall'^Tracks_Dense_1/StatefulPartitionedCall&^Tracks_Latent/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2T
(Final_Prediction/StatefulPartitionedCall(Final_Prediction/StatefulPartitionedCall2V
)Playlists_Dense_1/StatefulPartitionedCall)Playlists_Dense_1/StatefulPartitionedCall2V
)Playlists_Dense_2/StatefulPartitionedCall)Playlists_Dense_2/StatefulPartitionedCall2V
)Playlists_Dense_3/StatefulPartitionedCall)Playlists_Dense_3/StatefulPartitionedCall2T
(Playlists_Latent/StatefulPartitionedCall(Playlists_Latent/StatefulPartitionedCall2L
$Prediction_1/StatefulPartitionedCall$Prediction_1/StatefulPartitionedCall2L
$Prediction_2/StatefulPartitionedCall$Prediction_2/StatefulPartitionedCall2L
$Prediction_3/StatefulPartitionedCall$Prediction_3/StatefulPartitionedCall2L
$Prediction_4/StatefulPartitionedCall$Prediction_4/StatefulPartitionedCall2P
&Tracks_Dense_1/StatefulPartitionedCall&Tracks_Dense_1/StatefulPartitionedCall2N
%Tracks_Latent/StatefulPartitionedCall%Tracks_Latent/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

,__inference_Prediction_1_layer_call_fn_41462

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_1_layer_call_and_return_conditional_losses_40398p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
2
!__inference__traced_restore_42043
file_prefix=
)assignvariableop_playlists_dense_1_kernel:
8
)assignvariableop_1_playlists_dense_1_bias:	?
+assignvariableop_2_playlists_dense_2_kernel:
8
)assignvariableop_3_playlists_dense_2_bias:	?
+assignvariableop_4_playlists_dense_3_kernel:
8
)assignvariableop_5_playlists_dense_3_bias:	;
(assignvariableop_6_tracks_dense_1_kernel:	5
&assignvariableop_7_tracks_dense_1_bias:	=
*assignvariableop_8_playlists_latent_kernel:	@6
(assignvariableop_9_playlists_latent_bias:@;
(assignvariableop_10_tracks_latent_kernel:	@4
&assignvariableop_11_tracks_latent_bias:@;
'assignvariableop_12_prediction_1_kernel:
4
%assignvariableop_13_prediction_1_bias:	;
'assignvariableop_14_prediction_2_kernel:
4
%assignvariableop_15_prediction_2_bias:	;
'assignvariableop_16_prediction_3_kernel:
4
%assignvariableop_17_prediction_3_bias:	;
'assignvariableop_18_prediction_4_kernel:
4
%assignvariableop_19_prediction_4_bias:	>
+assignvariableop_20_final_prediction_kernel:	7
)assignvariableop_21_final_prediction_bias:(
assignvariableop_22_nadam_iter:	 *
 assignvariableop_23_nadam_beta_1: *
 assignvariableop_24_nadam_beta_2: )
assignvariableop_25_nadam_decay: 1
'assignvariableop_26_nadam_learning_rate: 2
(assignvariableop_27_nadam_momentum_cache: #
assignvariableop_28_total: #
assignvariableop_29_count: %
assignvariableop_30_total_1: %
assignvariableop_31_count_1: H
4assignvariableop_32_nadam_playlists_dense_1_kernel_m:
A
2assignvariableop_33_nadam_playlists_dense_1_bias_m:	H
4assignvariableop_34_nadam_playlists_dense_2_kernel_m:
A
2assignvariableop_35_nadam_playlists_dense_2_bias_m:	H
4assignvariableop_36_nadam_playlists_dense_3_kernel_m:
A
2assignvariableop_37_nadam_playlists_dense_3_bias_m:	D
1assignvariableop_38_nadam_tracks_dense_1_kernel_m:	>
/assignvariableop_39_nadam_tracks_dense_1_bias_m:	F
3assignvariableop_40_nadam_playlists_latent_kernel_m:	@?
1assignvariableop_41_nadam_playlists_latent_bias_m:@C
0assignvariableop_42_nadam_tracks_latent_kernel_m:	@<
.assignvariableop_43_nadam_tracks_latent_bias_m:@C
/assignvariableop_44_nadam_prediction_1_kernel_m:
<
-assignvariableop_45_nadam_prediction_1_bias_m:	C
/assignvariableop_46_nadam_prediction_2_kernel_m:
<
-assignvariableop_47_nadam_prediction_2_bias_m:	C
/assignvariableop_48_nadam_prediction_3_kernel_m:
<
-assignvariableop_49_nadam_prediction_3_bias_m:	C
/assignvariableop_50_nadam_prediction_4_kernel_m:
<
-assignvariableop_51_nadam_prediction_4_bias_m:	F
3assignvariableop_52_nadam_final_prediction_kernel_m:	?
1assignvariableop_53_nadam_final_prediction_bias_m:H
4assignvariableop_54_nadam_playlists_dense_1_kernel_v:
A
2assignvariableop_55_nadam_playlists_dense_1_bias_v:	H
4assignvariableop_56_nadam_playlists_dense_2_kernel_v:
A
2assignvariableop_57_nadam_playlists_dense_2_bias_v:	H
4assignvariableop_58_nadam_playlists_dense_3_kernel_v:
A
2assignvariableop_59_nadam_playlists_dense_3_bias_v:	D
1assignvariableop_60_nadam_tracks_dense_1_kernel_v:	>
/assignvariableop_61_nadam_tracks_dense_1_bias_v:	F
3assignvariableop_62_nadam_playlists_latent_kernel_v:	@?
1assignvariableop_63_nadam_playlists_latent_bias_v:@C
0assignvariableop_64_nadam_tracks_latent_kernel_v:	@<
.assignvariableop_65_nadam_tracks_latent_bias_v:@C
/assignvariableop_66_nadam_prediction_1_kernel_v:
<
-assignvariableop_67_nadam_prediction_1_bias_v:	C
/assignvariableop_68_nadam_prediction_2_kernel_v:
<
-assignvariableop_69_nadam_prediction_2_bias_v:	C
/assignvariableop_70_nadam_prediction_3_kernel_v:
<
-assignvariableop_71_nadam_prediction_3_bias_v:	C
/assignvariableop_72_nadam_prediction_4_kernel_v:
<
-assignvariableop_73_nadam_prediction_4_bias_v:	F
3assignvariableop_74_nadam_final_prediction_kernel_v:	?
1assignvariableop_75_nadam_final_prediction_bias_v:
identity_77¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_8¢AssignVariableOp_9+
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*©*
value*B*MB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*¯
value¥B¢MB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¢
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ê
_output_shapes·
´:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*[
dtypesQ
O2M	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp)assignvariableop_playlists_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp)assignvariableop_1_playlists_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp+assignvariableop_2_playlists_dense_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp)assignvariableop_3_playlists_dense_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp+assignvariableop_4_playlists_dense_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp)assignvariableop_5_playlists_dense_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp(assignvariableop_6_tracks_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp&assignvariableop_7_tracks_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp*assignvariableop_8_playlists_latent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp(assignvariableop_9_playlists_latent_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp(assignvariableop_10_tracks_latent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp&assignvariableop_11_tracks_latent_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp'assignvariableop_12_prediction_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp%assignvariableop_13_prediction_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp'assignvariableop_14_prediction_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp%assignvariableop_15_prediction_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp'assignvariableop_16_prediction_3_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp%assignvariableop_17_prediction_3_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp'assignvariableop_18_prediction_4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp%assignvariableop_19_prediction_4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp+assignvariableop_20_final_prediction_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp)assignvariableop_21_final_prediction_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_nadam_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp assignvariableop_23_nadam_beta_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp assignvariableop_24_nadam_beta_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_nadam_decayIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp'assignvariableop_26_nadam_learning_rateIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp(assignvariableop_27_nadam_momentum_cacheIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOpassignvariableop_30_total_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOpassignvariableop_31_count_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_32AssignVariableOp4assignvariableop_32_nadam_playlists_dense_1_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_33AssignVariableOp2assignvariableop_33_nadam_playlists_dense_1_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_34AssignVariableOp4assignvariableop_34_nadam_playlists_dense_2_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_35AssignVariableOp2assignvariableop_35_nadam_playlists_dense_2_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_36AssignVariableOp4assignvariableop_36_nadam_playlists_dense_3_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_37AssignVariableOp2assignvariableop_37_nadam_playlists_dense_3_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_38AssignVariableOp1assignvariableop_38_nadam_tracks_dense_1_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_39AssignVariableOp/assignvariableop_39_nadam_tracks_dense_1_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_40AssignVariableOp3assignvariableop_40_nadam_playlists_latent_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_41AssignVariableOp1assignvariableop_41_nadam_playlists_latent_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_42AssignVariableOp0assignvariableop_42_nadam_tracks_latent_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp.assignvariableop_43_nadam_tracks_latent_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_44AssignVariableOp/assignvariableop_44_nadam_prediction_1_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp-assignvariableop_45_nadam_prediction_1_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_46AssignVariableOp/assignvariableop_46_nadam_prediction_2_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp-assignvariableop_47_nadam_prediction_2_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_48AssignVariableOp/assignvariableop_48_nadam_prediction_3_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp-assignvariableop_49_nadam_prediction_3_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_50AssignVariableOp/assignvariableop_50_nadam_prediction_4_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp-assignvariableop_51_nadam_prediction_4_bias_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_52AssignVariableOp3assignvariableop_52_nadam_final_prediction_kernel_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_53AssignVariableOp1assignvariableop_53_nadam_final_prediction_bias_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_54AssignVariableOp4assignvariableop_54_nadam_playlists_dense_1_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_55AssignVariableOp2assignvariableop_55_nadam_playlists_dense_1_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_56AssignVariableOp4assignvariableop_56_nadam_playlists_dense_2_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_57AssignVariableOp2assignvariableop_57_nadam_playlists_dense_2_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_58AssignVariableOp4assignvariableop_58_nadam_playlists_dense_3_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_59AssignVariableOp2assignvariableop_59_nadam_playlists_dense_3_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_60AssignVariableOp1assignvariableop_60_nadam_tracks_dense_1_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_61AssignVariableOp/assignvariableop_61_nadam_tracks_dense_1_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_62AssignVariableOp3assignvariableop_62_nadam_playlists_latent_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_63AssignVariableOp1assignvariableop_63_nadam_playlists_latent_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_64AssignVariableOp0assignvariableop_64_nadam_tracks_latent_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_65AssignVariableOp.assignvariableop_65_nadam_tracks_latent_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_66AssignVariableOp/assignvariableop_66_nadam_prediction_1_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOp-assignvariableop_67_nadam_prediction_1_bias_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_68AssignVariableOp/assignvariableop_68_nadam_prediction_2_kernel_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp-assignvariableop_69_nadam_prediction_2_bias_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_70AssignVariableOp/assignvariableop_70_nadam_prediction_3_kernel_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOp-assignvariableop_71_nadam_prediction_3_bias_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_72AssignVariableOp/assignvariableop_72_nadam_prediction_4_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_73AssignVariableOp-assignvariableop_73_nadam_prediction_4_bias_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_74AssignVariableOp3assignvariableop_74_nadam_final_prediction_kernel_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_75AssignVariableOp1assignvariableop_75_nadam_final_prediction_bias_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ×
Identity_76Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_77IdentityIdentity_76:output:0^NoOp_1*
T0*
_output_shapes
: Ä
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_77Identity_77:output:0*¯
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¨

ü
I__inference_Tracks_Dense_1_layer_call_and_return_conditional_losses_41400

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯


L__inference_Playlists_Dense_2_layer_call_and_return_conditional_losses_41360

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
{
Q__inference_Concatenated_Encodings_layer_call_and_return_conditional_losses_40385

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
£

ú
H__inference_Tracks_Latent_layer_call_and_return_conditional_losses_40372

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª

û
G__inference_Prediction_3_layer_call_and_return_conditional_losses_40432

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
j
N__inference_Platlists_Flattened_layer_call_and_return_conditional_losses_41320

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¯


L__inference_Playlists_Dense_2_layer_call_and_return_conditional_losses_40304

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯


L__inference_Playlists_Dense_1_layer_call_and_return_conditional_losses_41340

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

ý
K__inference_Playlists_Latent_layer_call_and_return_conditional_losses_41420

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó

0__inference_Final_Prediction_layer_call_fn_41542

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Final_Prediction_layer_call_and_return_conditional_losses_40466o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯


L__inference_Playlists_Dense_3_layer_call_and_return_conditional_losses_40338

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó

0__inference_Playlists_Latent_layer_call_fn_41409

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Playlists_Latent_layer_call_and_return_conditional_losses_40355o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

,__inference_Prediction_4_layer_call_fn_41522

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_4_layer_call_and_return_conditional_losses_40449p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²E
¦
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_40473

inputs
inputs_1+
playlists_dense_1_40288:
&
playlists_dense_1_40290:	+
playlists_dense_2_40305:
&
playlists_dense_2_40307:	'
tracks_dense_1_40322:	#
tracks_dense_1_40324:	+
playlists_dense_3_40339:
&
playlists_dense_3_40341:	)
playlists_latent_40356:	@$
playlists_latent_40358:@&
tracks_latent_40373:	@!
tracks_latent_40375:@&
prediction_1_40399:
!
prediction_1_40401:	&
prediction_2_40416:
!
prediction_2_40418:	&
prediction_3_40433:
!
prediction_3_40435:	&
prediction_4_40450:
!
prediction_4_40452:	)
final_prediction_40467:	$
final_prediction_40469:
identity¢(Final_Prediction/StatefulPartitionedCall¢)Playlists_Dense_1/StatefulPartitionedCall¢)Playlists_Dense_2/StatefulPartitionedCall¢)Playlists_Dense_3/StatefulPartitionedCall¢(Playlists_Latent/StatefulPartitionedCall¢$Prediction_1/StatefulPartitionedCall¢$Prediction_2/StatefulPartitionedCall¢$Prediction_3/StatefulPartitionedCall¢$Prediction_4/StatefulPartitionedCall¢&Tracks_Dense_1/StatefulPartitionedCall¢%Tracks_Latent/StatefulPartitionedCallÎ
#Platlists_Flattened/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Platlists_Flattened_layer_call_and_return_conditional_losses_40274¸
)Playlists_Dense_1/StatefulPartitionedCallStatefulPartitionedCall,Platlists_Flattened/PartitionedCall:output:0playlists_dense_1_40288playlists_dense_1_40290*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Playlists_Dense_1_layer_call_and_return_conditional_losses_40287¾
)Playlists_Dense_2/StatefulPartitionedCallStatefulPartitionedCall2Playlists_Dense_1/StatefulPartitionedCall:output:0playlists_dense_2_40305playlists_dense_2_40307*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Playlists_Dense_2_layer_call_and_return_conditional_losses_40304
&Tracks_Dense_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1tracks_dense_1_40322tracks_dense_1_40324*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Tracks_Dense_1_layer_call_and_return_conditional_losses_40321¾
)Playlists_Dense_3/StatefulPartitionedCallStatefulPartitionedCall2Playlists_Dense_2/StatefulPartitionedCall:output:0playlists_dense_3_40339playlists_dense_3_40341*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Playlists_Dense_3_layer_call_and_return_conditional_losses_40338¹
(Playlists_Latent/StatefulPartitionedCallStatefulPartitionedCall2Playlists_Dense_3/StatefulPartitionedCall:output:0playlists_latent_40356playlists_latent_40358*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Playlists_Latent_layer_call_and_return_conditional_losses_40355ª
%Tracks_Latent/StatefulPartitionedCallStatefulPartitionedCall/Tracks_Dense_1/StatefulPartitionedCall:output:0tracks_latent_40373tracks_latent_40375*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Tracks_Latent_layer_call_and_return_conditional_losses_40372°
&Concatenated_Encodings/PartitionedCallPartitionedCall1Playlists_Latent/StatefulPartitionedCall:output:0.Tracks_Latent/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Concatenated_Encodings_layer_call_and_return_conditional_losses_40385§
$Prediction_1/StatefulPartitionedCallStatefulPartitionedCall/Concatenated_Encodings/PartitionedCall:output:0prediction_1_40399prediction_1_40401*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_1_layer_call_and_return_conditional_losses_40398¥
$Prediction_2/StatefulPartitionedCallStatefulPartitionedCall-Prediction_1/StatefulPartitionedCall:output:0prediction_2_40416prediction_2_40418*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_2_layer_call_and_return_conditional_losses_40415¥
$Prediction_3/StatefulPartitionedCallStatefulPartitionedCall-Prediction_2/StatefulPartitionedCall:output:0prediction_3_40433prediction_3_40435*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_3_layer_call_and_return_conditional_losses_40432¥
$Prediction_4/StatefulPartitionedCallStatefulPartitionedCall-Prediction_3/StatefulPartitionedCall:output:0prediction_4_40450prediction_4_40452*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_4_layer_call_and_return_conditional_losses_40449´
(Final_Prediction/StatefulPartitionedCallStatefulPartitionedCall-Prediction_4/StatefulPartitionedCall:output:0final_prediction_40467final_prediction_40469*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Final_Prediction_layer_call_and_return_conditional_losses_40466
IdentityIdentity1Final_Prediction/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^Final_Prediction/StatefulPartitionedCall*^Playlists_Dense_1/StatefulPartitionedCall*^Playlists_Dense_2/StatefulPartitionedCall*^Playlists_Dense_3/StatefulPartitionedCall)^Playlists_Latent/StatefulPartitionedCall%^Prediction_1/StatefulPartitionedCall%^Prediction_2/StatefulPartitionedCall%^Prediction_3/StatefulPartitionedCall%^Prediction_4/StatefulPartitionedCall'^Tracks_Dense_1/StatefulPartitionedCall&^Tracks_Latent/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2T
(Final_Prediction/StatefulPartitionedCall(Final_Prediction/StatefulPartitionedCall2V
)Playlists_Dense_1/StatefulPartitionedCall)Playlists_Dense_1/StatefulPartitionedCall2V
)Playlists_Dense_2/StatefulPartitionedCall)Playlists_Dense_2/StatefulPartitionedCall2V
)Playlists_Dense_3/StatefulPartitionedCall)Playlists_Dense_3/StatefulPartitionedCall2T
(Playlists_Latent/StatefulPartitionedCall(Playlists_Latent/StatefulPartitionedCall2L
$Prediction_1/StatefulPartitionedCall$Prediction_1/StatefulPartitionedCall2L
$Prediction_2/StatefulPartitionedCall$Prediction_2/StatefulPartitionedCall2L
$Prediction_3/StatefulPartitionedCall$Prediction_3/StatefulPartitionedCall2L
$Prediction_4/StatefulPartitionedCall$Prediction_4/StatefulPartitionedCall2P
&Tracks_Dense_1/StatefulPartitionedCall&Tracks_Dense_1/StatefulPartitionedCall2N
%Tracks_Latent/StatefulPartitionedCall%Tracks_Latent/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯


L__inference_Playlists_Dense_1_layer_call_and_return_conditional_losses_40287

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
b
6__inference_Concatenated_Encodings_layer_call_fn_41446
inputs_0
inputs_1
identityÊ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Concatenated_Encodings_layer_call_and_return_conditional_losses_40385a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/1
Ï

,__inference_Prediction_3_layer_call_fn_41502

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_3_layer_call_and_return_conditional_losses_40432p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª

û
G__inference_Prediction_4_layer_call_and_return_conditional_losses_41533

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

ý
K__inference_Playlists_Latent_layer_call_and_return_conditional_losses_40355

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í

-__inference_Tracks_Latent_layer_call_fn_41429

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Tracks_Latent_layer_call_and_return_conditional_losses_40372o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯


L__inference_Playlists_Dense_3_layer_call_and_return_conditional_losses_41380

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ªp
ÿ
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_41171
inputs_0
inputs_1D
0playlists_dense_1_matmul_readvariableop_resource:
@
1playlists_dense_1_biasadd_readvariableop_resource:	D
0playlists_dense_2_matmul_readvariableop_resource:
@
1playlists_dense_2_biasadd_readvariableop_resource:	@
-tracks_dense_1_matmul_readvariableop_resource:	=
.tracks_dense_1_biasadd_readvariableop_resource:	D
0playlists_dense_3_matmul_readvariableop_resource:
@
1playlists_dense_3_biasadd_readvariableop_resource:	B
/playlists_latent_matmul_readvariableop_resource:	@>
0playlists_latent_biasadd_readvariableop_resource:@?
,tracks_latent_matmul_readvariableop_resource:	@;
-tracks_latent_biasadd_readvariableop_resource:@?
+prediction_1_matmul_readvariableop_resource:
;
,prediction_1_biasadd_readvariableop_resource:	?
+prediction_2_matmul_readvariableop_resource:
;
,prediction_2_biasadd_readvariableop_resource:	?
+prediction_3_matmul_readvariableop_resource:
;
,prediction_3_biasadd_readvariableop_resource:	?
+prediction_4_matmul_readvariableop_resource:
;
,prediction_4_biasadd_readvariableop_resource:	B
/final_prediction_matmul_readvariableop_resource:	>
0final_prediction_biasadd_readvariableop_resource:
identity¢'Final_Prediction/BiasAdd/ReadVariableOp¢&Final_Prediction/MatMul/ReadVariableOp¢(Playlists_Dense_1/BiasAdd/ReadVariableOp¢'Playlists_Dense_1/MatMul/ReadVariableOp¢(Playlists_Dense_2/BiasAdd/ReadVariableOp¢'Playlists_Dense_2/MatMul/ReadVariableOp¢(Playlists_Dense_3/BiasAdd/ReadVariableOp¢'Playlists_Dense_3/MatMul/ReadVariableOp¢'Playlists_Latent/BiasAdd/ReadVariableOp¢&Playlists_Latent/MatMul/ReadVariableOp¢#Prediction_1/BiasAdd/ReadVariableOp¢"Prediction_1/MatMul/ReadVariableOp¢#Prediction_2/BiasAdd/ReadVariableOp¢"Prediction_2/MatMul/ReadVariableOp¢#Prediction_3/BiasAdd/ReadVariableOp¢"Prediction_3/MatMul/ReadVariableOp¢#Prediction_4/BiasAdd/ReadVariableOp¢"Prediction_4/MatMul/ReadVariableOp¢%Tracks_Dense_1/BiasAdd/ReadVariableOp¢$Tracks_Dense_1/MatMul/ReadVariableOp¢$Tracks_Latent/BiasAdd/ReadVariableOp¢#Tracks_Latent/MatMul/ReadVariableOpj
Platlists_Flattened/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
Platlists_Flattened/ReshapeReshapeinputs_0"Platlists_Flattened/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'Playlists_Dense_1/MatMul/ReadVariableOpReadVariableOp0playlists_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¬
Playlists_Dense_1/MatMulMatMul$Platlists_Flattened/Reshape:output:0/Playlists_Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(Playlists_Dense_1/BiasAdd/ReadVariableOpReadVariableOp1playlists_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
Playlists_Dense_1/BiasAddBiasAdd"Playlists_Dense_1/MatMul:product:00Playlists_Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
Playlists_Dense_1/ReluRelu"Playlists_Dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'Playlists_Dense_2/MatMul/ReadVariableOpReadVariableOp0playlists_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¬
Playlists_Dense_2/MatMulMatMul$Playlists_Dense_1/Relu:activations:0/Playlists_Dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(Playlists_Dense_2/BiasAdd/ReadVariableOpReadVariableOp1playlists_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
Playlists_Dense_2/BiasAddBiasAdd"Playlists_Dense_2/MatMul:product:00Playlists_Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
Playlists_Dense_2/ReluRelu"Playlists_Dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$Tracks_Dense_1/MatMul/ReadVariableOpReadVariableOp-tracks_dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
Tracks_Dense_1/MatMulMatMulinputs_1,Tracks_Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%Tracks_Dense_1/BiasAdd/ReadVariableOpReadVariableOp.tracks_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¤
Tracks_Dense_1/BiasAddBiasAddTracks_Dense_1/MatMul:product:0-Tracks_Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
Tracks_Dense_1/ReluReluTracks_Dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'Playlists_Dense_3/MatMul/ReadVariableOpReadVariableOp0playlists_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¬
Playlists_Dense_3/MatMulMatMul$Playlists_Dense_2/Relu:activations:0/Playlists_Dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(Playlists_Dense_3/BiasAdd/ReadVariableOpReadVariableOp1playlists_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
Playlists_Dense_3/BiasAddBiasAdd"Playlists_Dense_3/MatMul:product:00Playlists_Dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
Playlists_Dense_3/ReluRelu"Playlists_Dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&Playlists_Latent/MatMul/ReadVariableOpReadVariableOp/playlists_latent_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0©
Playlists_Latent/MatMulMatMul$Playlists_Dense_3/Relu:activations:0.Playlists_Latent/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'Playlists_Latent/BiasAdd/ReadVariableOpReadVariableOp0playlists_latent_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0©
Playlists_Latent/BiasAddBiasAdd!Playlists_Latent/MatMul:product:0/Playlists_Latent/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
Playlists_Latent/ReluRelu!Playlists_Latent/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
#Tracks_Latent/MatMul/ReadVariableOpReadVariableOp,tracks_latent_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0 
Tracks_Latent/MatMulMatMul!Tracks_Dense_1/Relu:activations:0+Tracks_Latent/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$Tracks_Latent/BiasAdd/ReadVariableOpReadVariableOp-tracks_latent_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0 
Tracks_Latent/BiasAddBiasAddTracks_Latent/MatMul:product:0,Tracks_Latent/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
Tracks_Latent/ReluReluTracks_Latent/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
"Concatenated_Encodings/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ù
Concatenated_Encodings/concatConcatV2#Playlists_Latent/Relu:activations:0 Tracks_Latent/Relu:activations:0+Concatenated_Encodings/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Prediction_1/MatMul/ReadVariableOpReadVariableOp+prediction_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¤
Prediction_1/MatMulMatMul&Concatenated_Encodings/concat:output:0*Prediction_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Prediction_1/BiasAdd/ReadVariableOpReadVariableOp,prediction_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
Prediction_1/BiasAddBiasAddPrediction_1/MatMul:product:0+Prediction_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
Prediction_1/ReluReluPrediction_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Prediction_2/MatMul/ReadVariableOpReadVariableOp+prediction_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
Prediction_2/MatMulMatMulPrediction_1/Relu:activations:0*Prediction_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Prediction_2/BiasAdd/ReadVariableOpReadVariableOp,prediction_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
Prediction_2/BiasAddBiasAddPrediction_2/MatMul:product:0+Prediction_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
Prediction_2/ReluReluPrediction_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Prediction_3/MatMul/ReadVariableOpReadVariableOp+prediction_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
Prediction_3/MatMulMatMulPrediction_2/Relu:activations:0*Prediction_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Prediction_3/BiasAdd/ReadVariableOpReadVariableOp,prediction_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
Prediction_3/BiasAddBiasAddPrediction_3/MatMul:product:0+Prediction_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
Prediction_3/ReluReluPrediction_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Prediction_4/MatMul/ReadVariableOpReadVariableOp+prediction_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
Prediction_4/MatMulMatMulPrediction_3/Relu:activations:0*Prediction_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Prediction_4/BiasAdd/ReadVariableOpReadVariableOp,prediction_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
Prediction_4/BiasAddBiasAddPrediction_4/MatMul:product:0+Prediction_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
Prediction_4/ReluReluPrediction_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&Final_Prediction/MatMul/ReadVariableOpReadVariableOp/final_prediction_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¤
Final_Prediction/MatMulMatMulPrediction_4/Relu:activations:0.Final_Prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'Final_Prediction/BiasAdd/ReadVariableOpReadVariableOp0final_prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
Final_Prediction/BiasAddBiasAdd!Final_Prediction/MatMul:product:0/Final_Prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
Final_Prediction/SigmoidSigmoid!Final_Prediction/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
IdentityIdentityFinal_Prediction/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
NoOpNoOp(^Final_Prediction/BiasAdd/ReadVariableOp'^Final_Prediction/MatMul/ReadVariableOp)^Playlists_Dense_1/BiasAdd/ReadVariableOp(^Playlists_Dense_1/MatMul/ReadVariableOp)^Playlists_Dense_2/BiasAdd/ReadVariableOp(^Playlists_Dense_2/MatMul/ReadVariableOp)^Playlists_Dense_3/BiasAdd/ReadVariableOp(^Playlists_Dense_3/MatMul/ReadVariableOp(^Playlists_Latent/BiasAdd/ReadVariableOp'^Playlists_Latent/MatMul/ReadVariableOp$^Prediction_1/BiasAdd/ReadVariableOp#^Prediction_1/MatMul/ReadVariableOp$^Prediction_2/BiasAdd/ReadVariableOp#^Prediction_2/MatMul/ReadVariableOp$^Prediction_3/BiasAdd/ReadVariableOp#^Prediction_3/MatMul/ReadVariableOp$^Prediction_4/BiasAdd/ReadVariableOp#^Prediction_4/MatMul/ReadVariableOp&^Tracks_Dense_1/BiasAdd/ReadVariableOp%^Tracks_Dense_1/MatMul/ReadVariableOp%^Tracks_Latent/BiasAdd/ReadVariableOp$^Tracks_Latent/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2R
'Final_Prediction/BiasAdd/ReadVariableOp'Final_Prediction/BiasAdd/ReadVariableOp2P
&Final_Prediction/MatMul/ReadVariableOp&Final_Prediction/MatMul/ReadVariableOp2T
(Playlists_Dense_1/BiasAdd/ReadVariableOp(Playlists_Dense_1/BiasAdd/ReadVariableOp2R
'Playlists_Dense_1/MatMul/ReadVariableOp'Playlists_Dense_1/MatMul/ReadVariableOp2T
(Playlists_Dense_2/BiasAdd/ReadVariableOp(Playlists_Dense_2/BiasAdd/ReadVariableOp2R
'Playlists_Dense_2/MatMul/ReadVariableOp'Playlists_Dense_2/MatMul/ReadVariableOp2T
(Playlists_Dense_3/BiasAdd/ReadVariableOp(Playlists_Dense_3/BiasAdd/ReadVariableOp2R
'Playlists_Dense_3/MatMul/ReadVariableOp'Playlists_Dense_3/MatMul/ReadVariableOp2R
'Playlists_Latent/BiasAdd/ReadVariableOp'Playlists_Latent/BiasAdd/ReadVariableOp2P
&Playlists_Latent/MatMul/ReadVariableOp&Playlists_Latent/MatMul/ReadVariableOp2J
#Prediction_1/BiasAdd/ReadVariableOp#Prediction_1/BiasAdd/ReadVariableOp2H
"Prediction_1/MatMul/ReadVariableOp"Prediction_1/MatMul/ReadVariableOp2J
#Prediction_2/BiasAdd/ReadVariableOp#Prediction_2/BiasAdd/ReadVariableOp2H
"Prediction_2/MatMul/ReadVariableOp"Prediction_2/MatMul/ReadVariableOp2J
#Prediction_3/BiasAdd/ReadVariableOp#Prediction_3/BiasAdd/ReadVariableOp2H
"Prediction_3/MatMul/ReadVariableOp"Prediction_3/MatMul/ReadVariableOp2J
#Prediction_4/BiasAdd/ReadVariableOp#Prediction_4/BiasAdd/ReadVariableOp2H
"Prediction_4/MatMul/ReadVariableOp"Prediction_4/MatMul/ReadVariableOp2N
%Tracks_Dense_1/BiasAdd/ReadVariableOp%Tracks_Dense_1/BiasAdd/ReadVariableOp2L
$Tracks_Dense_1/MatMul/ReadVariableOp$Tracks_Dense_1/MatMul/ReadVariableOp2L
$Tracks_Latent/BiasAdd/ReadVariableOp$Tracks_Latent/BiasAdd/ReadVariableOp2J
#Tracks_Latent/MatMul/ReadVariableOp#Tracks_Latent/MatMul/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
óE
»
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_40917
playlist_inputs
audio_feature_inputs+
playlists_dense_1_40860:
&
playlists_dense_1_40862:	+
playlists_dense_2_40865:
&
playlists_dense_2_40867:	'
tracks_dense_1_40870:	#
tracks_dense_1_40872:	+
playlists_dense_3_40875:
&
playlists_dense_3_40877:	)
playlists_latent_40880:	@$
playlists_latent_40882:@&
tracks_latent_40885:	@!
tracks_latent_40887:@&
prediction_1_40891:
!
prediction_1_40893:	&
prediction_2_40896:
!
prediction_2_40898:	&
prediction_3_40901:
!
prediction_3_40903:	&
prediction_4_40906:
!
prediction_4_40908:	)
final_prediction_40911:	$
final_prediction_40913:
identity¢(Final_Prediction/StatefulPartitionedCall¢)Playlists_Dense_1/StatefulPartitionedCall¢)Playlists_Dense_2/StatefulPartitionedCall¢)Playlists_Dense_3/StatefulPartitionedCall¢(Playlists_Latent/StatefulPartitionedCall¢$Prediction_1/StatefulPartitionedCall¢$Prediction_2/StatefulPartitionedCall¢$Prediction_3/StatefulPartitionedCall¢$Prediction_4/StatefulPartitionedCall¢&Tracks_Dense_1/StatefulPartitionedCall¢%Tracks_Latent/StatefulPartitionedCall×
#Platlists_Flattened/PartitionedCallPartitionedCallplaylist_inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Platlists_Flattened_layer_call_and_return_conditional_losses_40274¸
)Playlists_Dense_1/StatefulPartitionedCallStatefulPartitionedCall,Platlists_Flattened/PartitionedCall:output:0playlists_dense_1_40860playlists_dense_1_40862*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Playlists_Dense_1_layer_call_and_return_conditional_losses_40287¾
)Playlists_Dense_2/StatefulPartitionedCallStatefulPartitionedCall2Playlists_Dense_1/StatefulPartitionedCall:output:0playlists_dense_2_40865playlists_dense_2_40867*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Playlists_Dense_2_layer_call_and_return_conditional_losses_40304
&Tracks_Dense_1/StatefulPartitionedCallStatefulPartitionedCallaudio_feature_inputstracks_dense_1_40870tracks_dense_1_40872*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Tracks_Dense_1_layer_call_and_return_conditional_losses_40321¾
)Playlists_Dense_3/StatefulPartitionedCallStatefulPartitionedCall2Playlists_Dense_2/StatefulPartitionedCall:output:0playlists_dense_3_40875playlists_dense_3_40877*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Playlists_Dense_3_layer_call_and_return_conditional_losses_40338¹
(Playlists_Latent/StatefulPartitionedCallStatefulPartitionedCall2Playlists_Dense_3/StatefulPartitionedCall:output:0playlists_latent_40880playlists_latent_40882*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Playlists_Latent_layer_call_and_return_conditional_losses_40355ª
%Tracks_Latent/StatefulPartitionedCallStatefulPartitionedCall/Tracks_Dense_1/StatefulPartitionedCall:output:0tracks_latent_40885tracks_latent_40887*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Tracks_Latent_layer_call_and_return_conditional_losses_40372°
&Concatenated_Encodings/PartitionedCallPartitionedCall1Playlists_Latent/StatefulPartitionedCall:output:0.Tracks_Latent/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Concatenated_Encodings_layer_call_and_return_conditional_losses_40385§
$Prediction_1/StatefulPartitionedCallStatefulPartitionedCall/Concatenated_Encodings/PartitionedCall:output:0prediction_1_40891prediction_1_40893*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_1_layer_call_and_return_conditional_losses_40398¥
$Prediction_2/StatefulPartitionedCallStatefulPartitionedCall-Prediction_1/StatefulPartitionedCall:output:0prediction_2_40896prediction_2_40898*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_2_layer_call_and_return_conditional_losses_40415¥
$Prediction_3/StatefulPartitionedCallStatefulPartitionedCall-Prediction_2/StatefulPartitionedCall:output:0prediction_3_40901prediction_3_40903*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_3_layer_call_and_return_conditional_losses_40432¥
$Prediction_4/StatefulPartitionedCallStatefulPartitionedCall-Prediction_3/StatefulPartitionedCall:output:0prediction_4_40906prediction_4_40908*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_4_layer_call_and_return_conditional_losses_40449´
(Final_Prediction/StatefulPartitionedCallStatefulPartitionedCall-Prediction_4/StatefulPartitionedCall:output:0final_prediction_40911final_prediction_40913*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Final_Prediction_layer_call_and_return_conditional_losses_40466
IdentityIdentity1Final_Prediction/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^Final_Prediction/StatefulPartitionedCall*^Playlists_Dense_1/StatefulPartitionedCall*^Playlists_Dense_2/StatefulPartitionedCall*^Playlists_Dense_3/StatefulPartitionedCall)^Playlists_Latent/StatefulPartitionedCall%^Prediction_1/StatefulPartitionedCall%^Prediction_2/StatefulPartitionedCall%^Prediction_3/StatefulPartitionedCall%^Prediction_4/StatefulPartitionedCall'^Tracks_Dense_1/StatefulPartitionedCall&^Tracks_Latent/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2T
(Final_Prediction/StatefulPartitionedCall(Final_Prediction/StatefulPartitionedCall2V
)Playlists_Dense_1/StatefulPartitionedCall)Playlists_Dense_1/StatefulPartitionedCall2V
)Playlists_Dense_2/StatefulPartitionedCall)Playlists_Dense_2/StatefulPartitionedCall2V
)Playlists_Dense_3/StatefulPartitionedCall)Playlists_Dense_3/StatefulPartitionedCall2T
(Playlists_Latent/StatefulPartitionedCall(Playlists_Latent/StatefulPartitionedCall2L
$Prediction_1/StatefulPartitionedCall$Prediction_1/StatefulPartitionedCall2L
$Prediction_2/StatefulPartitionedCall$Prediction_2/StatefulPartitionedCall2L
$Prediction_3/StatefulPartitionedCall$Prediction_3/StatefulPartitionedCall2L
$Prediction_4/StatefulPartitionedCall$Prediction_4/StatefulPartitionedCall2P
&Tracks_Dense_1/StatefulPartitionedCall&Tracks_Dense_1/StatefulPartitionedCall2N
%Tracks_Latent/StatefulPartitionedCall%Tracks_Latent/StatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)
_user_specified_namePlaylist_Inputs:]Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_nameAudio_Feature_Inputs
Ì
}
Q__inference_Concatenated_Encodings_layer_call_and_return_conditional_losses_41453
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/1
»
O
3__inference_Platlists_Flattened_layer_call_fn_41314

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Platlists_Flattened_layer_call_and_return_conditional_losses_40274a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
È
j
N__inference_Platlists_Flattened_layer_call_and_return_conditional_losses_40274

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
×
ï
1__inference_Track_Recommender_layer_call_fn_41085
inputs_0
inputs_1
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	@
	unknown_8:@
	unknown_9:	@

unknown_10:@

unknown_11:


unknown_12:	

unknown_13:


unknown_14:	

unknown_15:


unknown_16:	

unknown_17:


unknown_18:	

unknown_19:	

unknown_20:
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_40758o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
£

ú
H__inference_Tracks_Latent_layer_call_and_return_conditional_losses_41440

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª

û
G__inference_Prediction_2_layer_call_and_return_conditional_losses_40415

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
ï
1__inference_Track_Recommender_layer_call_fn_41035
inputs_0
inputs_1
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	@
	unknown_8:@
	unknown_9:	@

unknown_10:@

unknown_11:


unknown_12:	

unknown_13:


unknown_14:	

unknown_15:


unknown_16:	

unknown_17:


unknown_18:	

unknown_19:	

unknown_20:
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_40473o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
¥

ý
K__inference_Final_Prediction_layer_call_and_return_conditional_losses_40466

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
¡
1__inference_Playlists_Dense_2_layer_call_fn_41349

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Playlists_Dense_2_layer_call_and_return_conditional_losses_40304p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ªp
ÿ
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_41257
inputs_0
inputs_1D
0playlists_dense_1_matmul_readvariableop_resource:
@
1playlists_dense_1_biasadd_readvariableop_resource:	D
0playlists_dense_2_matmul_readvariableop_resource:
@
1playlists_dense_2_biasadd_readvariableop_resource:	@
-tracks_dense_1_matmul_readvariableop_resource:	=
.tracks_dense_1_biasadd_readvariableop_resource:	D
0playlists_dense_3_matmul_readvariableop_resource:
@
1playlists_dense_3_biasadd_readvariableop_resource:	B
/playlists_latent_matmul_readvariableop_resource:	@>
0playlists_latent_biasadd_readvariableop_resource:@?
,tracks_latent_matmul_readvariableop_resource:	@;
-tracks_latent_biasadd_readvariableop_resource:@?
+prediction_1_matmul_readvariableop_resource:
;
,prediction_1_biasadd_readvariableop_resource:	?
+prediction_2_matmul_readvariableop_resource:
;
,prediction_2_biasadd_readvariableop_resource:	?
+prediction_3_matmul_readvariableop_resource:
;
,prediction_3_biasadd_readvariableop_resource:	?
+prediction_4_matmul_readvariableop_resource:
;
,prediction_4_biasadd_readvariableop_resource:	B
/final_prediction_matmul_readvariableop_resource:	>
0final_prediction_biasadd_readvariableop_resource:
identity¢'Final_Prediction/BiasAdd/ReadVariableOp¢&Final_Prediction/MatMul/ReadVariableOp¢(Playlists_Dense_1/BiasAdd/ReadVariableOp¢'Playlists_Dense_1/MatMul/ReadVariableOp¢(Playlists_Dense_2/BiasAdd/ReadVariableOp¢'Playlists_Dense_2/MatMul/ReadVariableOp¢(Playlists_Dense_3/BiasAdd/ReadVariableOp¢'Playlists_Dense_3/MatMul/ReadVariableOp¢'Playlists_Latent/BiasAdd/ReadVariableOp¢&Playlists_Latent/MatMul/ReadVariableOp¢#Prediction_1/BiasAdd/ReadVariableOp¢"Prediction_1/MatMul/ReadVariableOp¢#Prediction_2/BiasAdd/ReadVariableOp¢"Prediction_2/MatMul/ReadVariableOp¢#Prediction_3/BiasAdd/ReadVariableOp¢"Prediction_3/MatMul/ReadVariableOp¢#Prediction_4/BiasAdd/ReadVariableOp¢"Prediction_4/MatMul/ReadVariableOp¢%Tracks_Dense_1/BiasAdd/ReadVariableOp¢$Tracks_Dense_1/MatMul/ReadVariableOp¢$Tracks_Latent/BiasAdd/ReadVariableOp¢#Tracks_Latent/MatMul/ReadVariableOpj
Platlists_Flattened/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
Platlists_Flattened/ReshapeReshapeinputs_0"Platlists_Flattened/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'Playlists_Dense_1/MatMul/ReadVariableOpReadVariableOp0playlists_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¬
Playlists_Dense_1/MatMulMatMul$Platlists_Flattened/Reshape:output:0/Playlists_Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(Playlists_Dense_1/BiasAdd/ReadVariableOpReadVariableOp1playlists_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
Playlists_Dense_1/BiasAddBiasAdd"Playlists_Dense_1/MatMul:product:00Playlists_Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
Playlists_Dense_1/ReluRelu"Playlists_Dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'Playlists_Dense_2/MatMul/ReadVariableOpReadVariableOp0playlists_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¬
Playlists_Dense_2/MatMulMatMul$Playlists_Dense_1/Relu:activations:0/Playlists_Dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(Playlists_Dense_2/BiasAdd/ReadVariableOpReadVariableOp1playlists_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
Playlists_Dense_2/BiasAddBiasAdd"Playlists_Dense_2/MatMul:product:00Playlists_Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
Playlists_Dense_2/ReluRelu"Playlists_Dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$Tracks_Dense_1/MatMul/ReadVariableOpReadVariableOp-tracks_dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
Tracks_Dense_1/MatMulMatMulinputs_1,Tracks_Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%Tracks_Dense_1/BiasAdd/ReadVariableOpReadVariableOp.tracks_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¤
Tracks_Dense_1/BiasAddBiasAddTracks_Dense_1/MatMul:product:0-Tracks_Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
Tracks_Dense_1/ReluReluTracks_Dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'Playlists_Dense_3/MatMul/ReadVariableOpReadVariableOp0playlists_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¬
Playlists_Dense_3/MatMulMatMul$Playlists_Dense_2/Relu:activations:0/Playlists_Dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(Playlists_Dense_3/BiasAdd/ReadVariableOpReadVariableOp1playlists_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0­
Playlists_Dense_3/BiasAddBiasAdd"Playlists_Dense_3/MatMul:product:00Playlists_Dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
Playlists_Dense_3/ReluRelu"Playlists_Dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&Playlists_Latent/MatMul/ReadVariableOpReadVariableOp/playlists_latent_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0©
Playlists_Latent/MatMulMatMul$Playlists_Dense_3/Relu:activations:0.Playlists_Latent/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'Playlists_Latent/BiasAdd/ReadVariableOpReadVariableOp0playlists_latent_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0©
Playlists_Latent/BiasAddBiasAdd!Playlists_Latent/MatMul:product:0/Playlists_Latent/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
Playlists_Latent/ReluRelu!Playlists_Latent/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
#Tracks_Latent/MatMul/ReadVariableOpReadVariableOp,tracks_latent_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0 
Tracks_Latent/MatMulMatMul!Tracks_Dense_1/Relu:activations:0+Tracks_Latent/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$Tracks_Latent/BiasAdd/ReadVariableOpReadVariableOp-tracks_latent_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0 
Tracks_Latent/BiasAddBiasAddTracks_Latent/MatMul:product:0,Tracks_Latent/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
Tracks_Latent/ReluReluTracks_Latent/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
"Concatenated_Encodings/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ù
Concatenated_Encodings/concatConcatV2#Playlists_Latent/Relu:activations:0 Tracks_Latent/Relu:activations:0+Concatenated_Encodings/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Prediction_1/MatMul/ReadVariableOpReadVariableOp+prediction_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¤
Prediction_1/MatMulMatMul&Concatenated_Encodings/concat:output:0*Prediction_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Prediction_1/BiasAdd/ReadVariableOpReadVariableOp,prediction_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
Prediction_1/BiasAddBiasAddPrediction_1/MatMul:product:0+Prediction_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
Prediction_1/ReluReluPrediction_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Prediction_2/MatMul/ReadVariableOpReadVariableOp+prediction_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
Prediction_2/MatMulMatMulPrediction_1/Relu:activations:0*Prediction_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Prediction_2/BiasAdd/ReadVariableOpReadVariableOp,prediction_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
Prediction_2/BiasAddBiasAddPrediction_2/MatMul:product:0+Prediction_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
Prediction_2/ReluReluPrediction_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Prediction_3/MatMul/ReadVariableOpReadVariableOp+prediction_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
Prediction_3/MatMulMatMulPrediction_2/Relu:activations:0*Prediction_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Prediction_3/BiasAdd/ReadVariableOpReadVariableOp,prediction_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
Prediction_3/BiasAddBiasAddPrediction_3/MatMul:product:0+Prediction_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
Prediction_3/ReluReluPrediction_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Prediction_4/MatMul/ReadVariableOpReadVariableOp+prediction_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
Prediction_4/MatMulMatMulPrediction_3/Relu:activations:0*Prediction_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Prediction_4/BiasAdd/ReadVariableOpReadVariableOp,prediction_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
Prediction_4/BiasAddBiasAddPrediction_4/MatMul:product:0+Prediction_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
Prediction_4/ReluReluPrediction_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&Final_Prediction/MatMul/ReadVariableOpReadVariableOp/final_prediction_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¤
Final_Prediction/MatMulMatMulPrediction_4/Relu:activations:0.Final_Prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'Final_Prediction/BiasAdd/ReadVariableOpReadVariableOp0final_prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
Final_Prediction/BiasAddBiasAdd!Final_Prediction/MatMul:product:0/Final_Prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
Final_Prediction/SigmoidSigmoid!Final_Prediction/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
IdentityIdentityFinal_Prediction/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
NoOpNoOp(^Final_Prediction/BiasAdd/ReadVariableOp'^Final_Prediction/MatMul/ReadVariableOp)^Playlists_Dense_1/BiasAdd/ReadVariableOp(^Playlists_Dense_1/MatMul/ReadVariableOp)^Playlists_Dense_2/BiasAdd/ReadVariableOp(^Playlists_Dense_2/MatMul/ReadVariableOp)^Playlists_Dense_3/BiasAdd/ReadVariableOp(^Playlists_Dense_3/MatMul/ReadVariableOp(^Playlists_Latent/BiasAdd/ReadVariableOp'^Playlists_Latent/MatMul/ReadVariableOp$^Prediction_1/BiasAdd/ReadVariableOp#^Prediction_1/MatMul/ReadVariableOp$^Prediction_2/BiasAdd/ReadVariableOp#^Prediction_2/MatMul/ReadVariableOp$^Prediction_3/BiasAdd/ReadVariableOp#^Prediction_3/MatMul/ReadVariableOp$^Prediction_4/BiasAdd/ReadVariableOp#^Prediction_4/MatMul/ReadVariableOp&^Tracks_Dense_1/BiasAdd/ReadVariableOp%^Tracks_Dense_1/MatMul/ReadVariableOp%^Tracks_Latent/BiasAdd/ReadVariableOp$^Tracks_Latent/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2R
'Final_Prediction/BiasAdd/ReadVariableOp'Final_Prediction/BiasAdd/ReadVariableOp2P
&Final_Prediction/MatMul/ReadVariableOp&Final_Prediction/MatMul/ReadVariableOp2T
(Playlists_Dense_1/BiasAdd/ReadVariableOp(Playlists_Dense_1/BiasAdd/ReadVariableOp2R
'Playlists_Dense_1/MatMul/ReadVariableOp'Playlists_Dense_1/MatMul/ReadVariableOp2T
(Playlists_Dense_2/BiasAdd/ReadVariableOp(Playlists_Dense_2/BiasAdd/ReadVariableOp2R
'Playlists_Dense_2/MatMul/ReadVariableOp'Playlists_Dense_2/MatMul/ReadVariableOp2T
(Playlists_Dense_3/BiasAdd/ReadVariableOp(Playlists_Dense_3/BiasAdd/ReadVariableOp2R
'Playlists_Dense_3/MatMul/ReadVariableOp'Playlists_Dense_3/MatMul/ReadVariableOp2R
'Playlists_Latent/BiasAdd/ReadVariableOp'Playlists_Latent/BiasAdd/ReadVariableOp2P
&Playlists_Latent/MatMul/ReadVariableOp&Playlists_Latent/MatMul/ReadVariableOp2J
#Prediction_1/BiasAdd/ReadVariableOp#Prediction_1/BiasAdd/ReadVariableOp2H
"Prediction_1/MatMul/ReadVariableOp"Prediction_1/MatMul/ReadVariableOp2J
#Prediction_2/BiasAdd/ReadVariableOp#Prediction_2/BiasAdd/ReadVariableOp2H
"Prediction_2/MatMul/ReadVariableOp"Prediction_2/MatMul/ReadVariableOp2J
#Prediction_3/BiasAdd/ReadVariableOp#Prediction_3/BiasAdd/ReadVariableOp2H
"Prediction_3/MatMul/ReadVariableOp"Prediction_3/MatMul/ReadVariableOp2J
#Prediction_4/BiasAdd/ReadVariableOp#Prediction_4/BiasAdd/ReadVariableOp2H
"Prediction_4/MatMul/ReadVariableOp"Prediction_4/MatMul/ReadVariableOp2N
%Tracks_Dense_1/BiasAdd/ReadVariableOp%Tracks_Dense_1/BiasAdd/ReadVariableOp2L
$Tracks_Dense_1/MatMul/ReadVariableOp$Tracks_Dense_1/MatMul/ReadVariableOp2L
$Tracks_Latent/BiasAdd/ReadVariableOp$Tracks_Latent/BiasAdd/ReadVariableOp2J
#Tracks_Latent/MatMul/ReadVariableOp#Tracks_Latent/MatMul/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ö
þ
 __inference__wrapped_model_40259
playlist_inputs
audio_feature_inputsV
Btrack_recommender_playlists_dense_1_matmul_readvariableop_resource:
R
Ctrack_recommender_playlists_dense_1_biasadd_readvariableop_resource:	V
Btrack_recommender_playlists_dense_2_matmul_readvariableop_resource:
R
Ctrack_recommender_playlists_dense_2_biasadd_readvariableop_resource:	R
?track_recommender_tracks_dense_1_matmul_readvariableop_resource:	O
@track_recommender_tracks_dense_1_biasadd_readvariableop_resource:	V
Btrack_recommender_playlists_dense_3_matmul_readvariableop_resource:
R
Ctrack_recommender_playlists_dense_3_biasadd_readvariableop_resource:	T
Atrack_recommender_playlists_latent_matmul_readvariableop_resource:	@P
Btrack_recommender_playlists_latent_biasadd_readvariableop_resource:@Q
>track_recommender_tracks_latent_matmul_readvariableop_resource:	@M
?track_recommender_tracks_latent_biasadd_readvariableop_resource:@Q
=track_recommender_prediction_1_matmul_readvariableop_resource:
M
>track_recommender_prediction_1_biasadd_readvariableop_resource:	Q
=track_recommender_prediction_2_matmul_readvariableop_resource:
M
>track_recommender_prediction_2_biasadd_readvariableop_resource:	Q
=track_recommender_prediction_3_matmul_readvariableop_resource:
M
>track_recommender_prediction_3_biasadd_readvariableop_resource:	Q
=track_recommender_prediction_4_matmul_readvariableop_resource:
M
>track_recommender_prediction_4_biasadd_readvariableop_resource:	T
Atrack_recommender_final_prediction_matmul_readvariableop_resource:	P
Btrack_recommender_final_prediction_biasadd_readvariableop_resource:
identity¢9Track_Recommender/Final_Prediction/BiasAdd/ReadVariableOp¢8Track_Recommender/Final_Prediction/MatMul/ReadVariableOp¢:Track_Recommender/Playlists_Dense_1/BiasAdd/ReadVariableOp¢9Track_Recommender/Playlists_Dense_1/MatMul/ReadVariableOp¢:Track_Recommender/Playlists_Dense_2/BiasAdd/ReadVariableOp¢9Track_Recommender/Playlists_Dense_2/MatMul/ReadVariableOp¢:Track_Recommender/Playlists_Dense_3/BiasAdd/ReadVariableOp¢9Track_Recommender/Playlists_Dense_3/MatMul/ReadVariableOp¢9Track_Recommender/Playlists_Latent/BiasAdd/ReadVariableOp¢8Track_Recommender/Playlists_Latent/MatMul/ReadVariableOp¢5Track_Recommender/Prediction_1/BiasAdd/ReadVariableOp¢4Track_Recommender/Prediction_1/MatMul/ReadVariableOp¢5Track_Recommender/Prediction_2/BiasAdd/ReadVariableOp¢4Track_Recommender/Prediction_2/MatMul/ReadVariableOp¢5Track_Recommender/Prediction_3/BiasAdd/ReadVariableOp¢4Track_Recommender/Prediction_3/MatMul/ReadVariableOp¢5Track_Recommender/Prediction_4/BiasAdd/ReadVariableOp¢4Track_Recommender/Prediction_4/MatMul/ReadVariableOp¢7Track_Recommender/Tracks_Dense_1/BiasAdd/ReadVariableOp¢6Track_Recommender/Tracks_Dense_1/MatMul/ReadVariableOp¢6Track_Recommender/Tracks_Latent/BiasAdd/ReadVariableOp¢5Track_Recommender/Tracks_Latent/MatMul/ReadVariableOp|
+Track_Recommender/Platlists_Flattened/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ²
-Track_Recommender/Platlists_Flattened/ReshapeReshapeplaylist_inputs4Track_Recommender/Platlists_Flattened/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9Track_Recommender/Playlists_Dense_1/MatMul/ReadVariableOpReadVariableOpBtrack_recommender_playlists_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0â
*Track_Recommender/Playlists_Dense_1/MatMulMatMul6Track_Recommender/Platlists_Flattened/Reshape:output:0ATrack_Recommender/Playlists_Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
:Track_Recommender/Playlists_Dense_1/BiasAdd/ReadVariableOpReadVariableOpCtrack_recommender_playlists_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ã
+Track_Recommender/Playlists_Dense_1/BiasAddBiasAdd4Track_Recommender/Playlists_Dense_1/MatMul:product:0BTrack_Recommender/Playlists_Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(Track_Recommender/Playlists_Dense_1/ReluRelu4Track_Recommender/Playlists_Dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9Track_Recommender/Playlists_Dense_2/MatMul/ReadVariableOpReadVariableOpBtrack_recommender_playlists_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0â
*Track_Recommender/Playlists_Dense_2/MatMulMatMul6Track_Recommender/Playlists_Dense_1/Relu:activations:0ATrack_Recommender/Playlists_Dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
:Track_Recommender/Playlists_Dense_2/BiasAdd/ReadVariableOpReadVariableOpCtrack_recommender_playlists_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ã
+Track_Recommender/Playlists_Dense_2/BiasAddBiasAdd4Track_Recommender/Playlists_Dense_2/MatMul:product:0BTrack_Recommender/Playlists_Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(Track_Recommender/Playlists_Dense_2/ReluRelu4Track_Recommender/Playlists_Dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
6Track_Recommender/Tracks_Dense_1/MatMul/ReadVariableOpReadVariableOp?track_recommender_tracks_dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0º
'Track_Recommender/Tracks_Dense_1/MatMulMatMulaudio_feature_inputs>Track_Recommender/Tracks_Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
7Track_Recommender/Tracks_Dense_1/BiasAdd/ReadVariableOpReadVariableOp@track_recommender_tracks_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ú
(Track_Recommender/Tracks_Dense_1/BiasAddBiasAdd1Track_Recommender/Tracks_Dense_1/MatMul:product:0?Track_Recommender/Tracks_Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%Track_Recommender/Tracks_Dense_1/ReluRelu1Track_Recommender/Tracks_Dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
9Track_Recommender/Playlists_Dense_3/MatMul/ReadVariableOpReadVariableOpBtrack_recommender_playlists_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0â
*Track_Recommender/Playlists_Dense_3/MatMulMatMul6Track_Recommender/Playlists_Dense_2/Relu:activations:0ATrack_Recommender/Playlists_Dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
:Track_Recommender/Playlists_Dense_3/BiasAdd/ReadVariableOpReadVariableOpCtrack_recommender_playlists_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ã
+Track_Recommender/Playlists_Dense_3/BiasAddBiasAdd4Track_Recommender/Playlists_Dense_3/MatMul:product:0BTrack_Recommender/Playlists_Dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(Track_Recommender/Playlists_Dense_3/ReluRelu4Track_Recommender/Playlists_Dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
8Track_Recommender/Playlists_Latent/MatMul/ReadVariableOpReadVariableOpAtrack_recommender_playlists_latent_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0ß
)Track_Recommender/Playlists_Latent/MatMulMatMul6Track_Recommender/Playlists_Dense_3/Relu:activations:0@Track_Recommender/Playlists_Latent/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¸
9Track_Recommender/Playlists_Latent/BiasAdd/ReadVariableOpReadVariableOpBtrack_recommender_playlists_latent_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ß
*Track_Recommender/Playlists_Latent/BiasAddBiasAdd3Track_Recommender/Playlists_Latent/MatMul:product:0ATrack_Recommender/Playlists_Latent/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'Track_Recommender/Playlists_Latent/ReluRelu3Track_Recommender/Playlists_Latent/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@µ
5Track_Recommender/Tracks_Latent/MatMul/ReadVariableOpReadVariableOp>track_recommender_tracks_latent_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0Ö
&Track_Recommender/Tracks_Latent/MatMulMatMul3Track_Recommender/Tracks_Dense_1/Relu:activations:0=Track_Recommender/Tracks_Latent/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@²
6Track_Recommender/Tracks_Latent/BiasAdd/ReadVariableOpReadVariableOp?track_recommender_tracks_latent_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
'Track_Recommender/Tracks_Latent/BiasAddBiasAdd0Track_Recommender/Tracks_Latent/MatMul:product:0>Track_Recommender/Tracks_Latent/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$Track_Recommender/Tracks_Latent/ReluRelu0Track_Recommender/Tracks_Latent/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@v
4Track_Recommender/Concatenated_Encodings/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¡
/Track_Recommender/Concatenated_Encodings/concatConcatV25Track_Recommender/Playlists_Latent/Relu:activations:02Track_Recommender/Tracks_Latent/Relu:activations:0=Track_Recommender/Concatenated_Encodings/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
4Track_Recommender/Prediction_1/MatMul/ReadVariableOpReadVariableOp=track_recommender_prediction_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ú
%Track_Recommender/Prediction_1/MatMulMatMul8Track_Recommender/Concatenated_Encodings/concat:output:0<Track_Recommender/Prediction_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
5Track_Recommender/Prediction_1/BiasAdd/ReadVariableOpReadVariableOp>track_recommender_prediction_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ô
&Track_Recommender/Prediction_1/BiasAddBiasAdd/Track_Recommender/Prediction_1/MatMul:product:0=Track_Recommender/Prediction_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Track_Recommender/Prediction_1/ReluRelu/Track_Recommender/Prediction_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
4Track_Recommender/Prediction_2/MatMul/ReadVariableOpReadVariableOp=track_recommender_prediction_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ó
%Track_Recommender/Prediction_2/MatMulMatMul1Track_Recommender/Prediction_1/Relu:activations:0<Track_Recommender/Prediction_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
5Track_Recommender/Prediction_2/BiasAdd/ReadVariableOpReadVariableOp>track_recommender_prediction_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ô
&Track_Recommender/Prediction_2/BiasAddBiasAdd/Track_Recommender/Prediction_2/MatMul:product:0=Track_Recommender/Prediction_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Track_Recommender/Prediction_2/ReluRelu/Track_Recommender/Prediction_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
4Track_Recommender/Prediction_3/MatMul/ReadVariableOpReadVariableOp=track_recommender_prediction_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ó
%Track_Recommender/Prediction_3/MatMulMatMul1Track_Recommender/Prediction_2/Relu:activations:0<Track_Recommender/Prediction_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
5Track_Recommender/Prediction_3/BiasAdd/ReadVariableOpReadVariableOp>track_recommender_prediction_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ô
&Track_Recommender/Prediction_3/BiasAddBiasAdd/Track_Recommender/Prediction_3/MatMul:product:0=Track_Recommender/Prediction_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Track_Recommender/Prediction_3/ReluRelu/Track_Recommender/Prediction_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
4Track_Recommender/Prediction_4/MatMul/ReadVariableOpReadVariableOp=track_recommender_prediction_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ó
%Track_Recommender/Prediction_4/MatMulMatMul1Track_Recommender/Prediction_3/Relu:activations:0<Track_Recommender/Prediction_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
5Track_Recommender/Prediction_4/BiasAdd/ReadVariableOpReadVariableOp>track_recommender_prediction_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ô
&Track_Recommender/Prediction_4/BiasAddBiasAdd/Track_Recommender/Prediction_4/MatMul:product:0=Track_Recommender/Prediction_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Track_Recommender/Prediction_4/ReluRelu/Track_Recommender/Prediction_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
8Track_Recommender/Final_Prediction/MatMul/ReadVariableOpReadVariableOpAtrack_recommender_final_prediction_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ú
)Track_Recommender/Final_Prediction/MatMulMatMul1Track_Recommender/Prediction_4/Relu:activations:0@Track_Recommender/Final_Prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
9Track_Recommender/Final_Prediction/BiasAdd/ReadVariableOpReadVariableOpBtrack_recommender_final_prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ß
*Track_Recommender/Final_Prediction/BiasAddBiasAdd3Track_Recommender/Final_Prediction/MatMul:product:0ATrack_Recommender/Final_Prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*Track_Recommender/Final_Prediction/SigmoidSigmoid3Track_Recommender/Final_Prediction/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
IdentityIdentity.Track_Recommender/Final_Prediction/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿

NoOpNoOp:^Track_Recommender/Final_Prediction/BiasAdd/ReadVariableOp9^Track_Recommender/Final_Prediction/MatMul/ReadVariableOp;^Track_Recommender/Playlists_Dense_1/BiasAdd/ReadVariableOp:^Track_Recommender/Playlists_Dense_1/MatMul/ReadVariableOp;^Track_Recommender/Playlists_Dense_2/BiasAdd/ReadVariableOp:^Track_Recommender/Playlists_Dense_2/MatMul/ReadVariableOp;^Track_Recommender/Playlists_Dense_3/BiasAdd/ReadVariableOp:^Track_Recommender/Playlists_Dense_3/MatMul/ReadVariableOp:^Track_Recommender/Playlists_Latent/BiasAdd/ReadVariableOp9^Track_Recommender/Playlists_Latent/MatMul/ReadVariableOp6^Track_Recommender/Prediction_1/BiasAdd/ReadVariableOp5^Track_Recommender/Prediction_1/MatMul/ReadVariableOp6^Track_Recommender/Prediction_2/BiasAdd/ReadVariableOp5^Track_Recommender/Prediction_2/MatMul/ReadVariableOp6^Track_Recommender/Prediction_3/BiasAdd/ReadVariableOp5^Track_Recommender/Prediction_3/MatMul/ReadVariableOp6^Track_Recommender/Prediction_4/BiasAdd/ReadVariableOp5^Track_Recommender/Prediction_4/MatMul/ReadVariableOp8^Track_Recommender/Tracks_Dense_1/BiasAdd/ReadVariableOp7^Track_Recommender/Tracks_Dense_1/MatMul/ReadVariableOp7^Track_Recommender/Tracks_Latent/BiasAdd/ReadVariableOp6^Track_Recommender/Tracks_Latent/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2v
9Track_Recommender/Final_Prediction/BiasAdd/ReadVariableOp9Track_Recommender/Final_Prediction/BiasAdd/ReadVariableOp2t
8Track_Recommender/Final_Prediction/MatMul/ReadVariableOp8Track_Recommender/Final_Prediction/MatMul/ReadVariableOp2x
:Track_Recommender/Playlists_Dense_1/BiasAdd/ReadVariableOp:Track_Recommender/Playlists_Dense_1/BiasAdd/ReadVariableOp2v
9Track_Recommender/Playlists_Dense_1/MatMul/ReadVariableOp9Track_Recommender/Playlists_Dense_1/MatMul/ReadVariableOp2x
:Track_Recommender/Playlists_Dense_2/BiasAdd/ReadVariableOp:Track_Recommender/Playlists_Dense_2/BiasAdd/ReadVariableOp2v
9Track_Recommender/Playlists_Dense_2/MatMul/ReadVariableOp9Track_Recommender/Playlists_Dense_2/MatMul/ReadVariableOp2x
:Track_Recommender/Playlists_Dense_3/BiasAdd/ReadVariableOp:Track_Recommender/Playlists_Dense_3/BiasAdd/ReadVariableOp2v
9Track_Recommender/Playlists_Dense_3/MatMul/ReadVariableOp9Track_Recommender/Playlists_Dense_3/MatMul/ReadVariableOp2v
9Track_Recommender/Playlists_Latent/BiasAdd/ReadVariableOp9Track_Recommender/Playlists_Latent/BiasAdd/ReadVariableOp2t
8Track_Recommender/Playlists_Latent/MatMul/ReadVariableOp8Track_Recommender/Playlists_Latent/MatMul/ReadVariableOp2n
5Track_Recommender/Prediction_1/BiasAdd/ReadVariableOp5Track_Recommender/Prediction_1/BiasAdd/ReadVariableOp2l
4Track_Recommender/Prediction_1/MatMul/ReadVariableOp4Track_Recommender/Prediction_1/MatMul/ReadVariableOp2n
5Track_Recommender/Prediction_2/BiasAdd/ReadVariableOp5Track_Recommender/Prediction_2/BiasAdd/ReadVariableOp2l
4Track_Recommender/Prediction_2/MatMul/ReadVariableOp4Track_Recommender/Prediction_2/MatMul/ReadVariableOp2n
5Track_Recommender/Prediction_3/BiasAdd/ReadVariableOp5Track_Recommender/Prediction_3/BiasAdd/ReadVariableOp2l
4Track_Recommender/Prediction_3/MatMul/ReadVariableOp4Track_Recommender/Prediction_3/MatMul/ReadVariableOp2n
5Track_Recommender/Prediction_4/BiasAdd/ReadVariableOp5Track_Recommender/Prediction_4/BiasAdd/ReadVariableOp2l
4Track_Recommender/Prediction_4/MatMul/ReadVariableOp4Track_Recommender/Prediction_4/MatMul/ReadVariableOp2r
7Track_Recommender/Tracks_Dense_1/BiasAdd/ReadVariableOp7Track_Recommender/Tracks_Dense_1/BiasAdd/ReadVariableOp2p
6Track_Recommender/Tracks_Dense_1/MatMul/ReadVariableOp6Track_Recommender/Tracks_Dense_1/MatMul/ReadVariableOp2p
6Track_Recommender/Tracks_Latent/BiasAdd/ReadVariableOp6Track_Recommender/Tracks_Latent/BiasAdd/ReadVariableOp2n
5Track_Recommender/Tracks_Latent/MatMul/ReadVariableOp5Track_Recommender/Tracks_Latent/MatMul/ReadVariableOp:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)
_user_specified_namePlaylist_Inputs:]Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_nameAudio_Feature_Inputs
Ö
ô
#__inference_signature_wrapper_41309
audio_feature_inputs
playlist_inputs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	@
	unknown_8:@
	unknown_9:	@

unknown_10:@

unknown_11:


unknown_12:	

unknown_13:


unknown_14:	

unknown_15:


unknown_16:	

unknown_17:


unknown_18:	

unknown_19:	

unknown_20:
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallplaylist_inputsaudio_feature_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_40259o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_nameAudio_Feature_Inputs:\X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)
_user_specified_namePlaylist_Inputs
ª

û
G__inference_Prediction_1_layer_call_and_return_conditional_losses_41473

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
óE
»
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_40979
playlist_inputs
audio_feature_inputs+
playlists_dense_1_40922:
&
playlists_dense_1_40924:	+
playlists_dense_2_40927:
&
playlists_dense_2_40929:	'
tracks_dense_1_40932:	#
tracks_dense_1_40934:	+
playlists_dense_3_40937:
&
playlists_dense_3_40939:	)
playlists_latent_40942:	@$
playlists_latent_40944:@&
tracks_latent_40947:	@!
tracks_latent_40949:@&
prediction_1_40953:
!
prediction_1_40955:	&
prediction_2_40958:
!
prediction_2_40960:	&
prediction_3_40963:
!
prediction_3_40965:	&
prediction_4_40968:
!
prediction_4_40970:	)
final_prediction_40973:	$
final_prediction_40975:
identity¢(Final_Prediction/StatefulPartitionedCall¢)Playlists_Dense_1/StatefulPartitionedCall¢)Playlists_Dense_2/StatefulPartitionedCall¢)Playlists_Dense_3/StatefulPartitionedCall¢(Playlists_Latent/StatefulPartitionedCall¢$Prediction_1/StatefulPartitionedCall¢$Prediction_2/StatefulPartitionedCall¢$Prediction_3/StatefulPartitionedCall¢$Prediction_4/StatefulPartitionedCall¢&Tracks_Dense_1/StatefulPartitionedCall¢%Tracks_Latent/StatefulPartitionedCall×
#Platlists_Flattened/PartitionedCallPartitionedCallplaylist_inputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_Platlists_Flattened_layer_call_and_return_conditional_losses_40274¸
)Playlists_Dense_1/StatefulPartitionedCallStatefulPartitionedCall,Platlists_Flattened/PartitionedCall:output:0playlists_dense_1_40922playlists_dense_1_40924*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Playlists_Dense_1_layer_call_and_return_conditional_losses_40287¾
)Playlists_Dense_2/StatefulPartitionedCallStatefulPartitionedCall2Playlists_Dense_1/StatefulPartitionedCall:output:0playlists_dense_2_40927playlists_dense_2_40929*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Playlists_Dense_2_layer_call_and_return_conditional_losses_40304
&Tracks_Dense_1/StatefulPartitionedCallStatefulPartitionedCallaudio_feature_inputstracks_dense_1_40932tracks_dense_1_40934*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Tracks_Dense_1_layer_call_and_return_conditional_losses_40321¾
)Playlists_Dense_3/StatefulPartitionedCallStatefulPartitionedCall2Playlists_Dense_2/StatefulPartitionedCall:output:0playlists_dense_3_40937playlists_dense_3_40939*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Playlists_Dense_3_layer_call_and_return_conditional_losses_40338¹
(Playlists_Latent/StatefulPartitionedCallStatefulPartitionedCall2Playlists_Dense_3/StatefulPartitionedCall:output:0playlists_latent_40942playlists_latent_40944*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Playlists_Latent_layer_call_and_return_conditional_losses_40355ª
%Tracks_Latent/StatefulPartitionedCallStatefulPartitionedCall/Tracks_Dense_1/StatefulPartitionedCall:output:0tracks_latent_40947tracks_latent_40949*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Tracks_Latent_layer_call_and_return_conditional_losses_40372°
&Concatenated_Encodings/PartitionedCallPartitionedCall1Playlists_Latent/StatefulPartitionedCall:output:0.Tracks_Latent/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_Concatenated_Encodings_layer_call_and_return_conditional_losses_40385§
$Prediction_1/StatefulPartitionedCallStatefulPartitionedCall/Concatenated_Encodings/PartitionedCall:output:0prediction_1_40953prediction_1_40955*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_1_layer_call_and_return_conditional_losses_40398¥
$Prediction_2/StatefulPartitionedCallStatefulPartitionedCall-Prediction_1/StatefulPartitionedCall:output:0prediction_2_40958prediction_2_40960*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_2_layer_call_and_return_conditional_losses_40415¥
$Prediction_3/StatefulPartitionedCallStatefulPartitionedCall-Prediction_2/StatefulPartitionedCall:output:0prediction_3_40963prediction_3_40965*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_3_layer_call_and_return_conditional_losses_40432¥
$Prediction_4/StatefulPartitionedCallStatefulPartitionedCall-Prediction_3/StatefulPartitionedCall:output:0prediction_4_40968prediction_4_40970*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Prediction_4_layer_call_and_return_conditional_losses_40449´
(Final_Prediction/StatefulPartitionedCallStatefulPartitionedCall-Prediction_4/StatefulPartitionedCall:output:0final_prediction_40973final_prediction_40975*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_Final_Prediction_layer_call_and_return_conditional_losses_40466
IdentityIdentity1Final_Prediction/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^Final_Prediction/StatefulPartitionedCall*^Playlists_Dense_1/StatefulPartitionedCall*^Playlists_Dense_2/StatefulPartitionedCall*^Playlists_Dense_3/StatefulPartitionedCall)^Playlists_Latent/StatefulPartitionedCall%^Prediction_1/StatefulPartitionedCall%^Prediction_2/StatefulPartitionedCall%^Prediction_3/StatefulPartitionedCall%^Prediction_4/StatefulPartitionedCall'^Tracks_Dense_1/StatefulPartitionedCall&^Tracks_Latent/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2T
(Final_Prediction/StatefulPartitionedCall(Final_Prediction/StatefulPartitionedCall2V
)Playlists_Dense_1/StatefulPartitionedCall)Playlists_Dense_1/StatefulPartitionedCall2V
)Playlists_Dense_2/StatefulPartitionedCall)Playlists_Dense_2/StatefulPartitionedCall2V
)Playlists_Dense_3/StatefulPartitionedCall)Playlists_Dense_3/StatefulPartitionedCall2T
(Playlists_Latent/StatefulPartitionedCall(Playlists_Latent/StatefulPartitionedCall2L
$Prediction_1/StatefulPartitionedCall$Prediction_1/StatefulPartitionedCall2L
$Prediction_2/StatefulPartitionedCall$Prediction_2/StatefulPartitionedCall2L
$Prediction_3/StatefulPartitionedCall$Prediction_3/StatefulPartitionedCall2L
$Prediction_4/StatefulPartitionedCall$Prediction_4/StatefulPartitionedCall2P
&Tracks_Dense_1/StatefulPartitionedCall&Tracks_Dense_1/StatefulPartitionedCall2N
%Tracks_Latent/StatefulPartitionedCall%Tracks_Latent/StatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)
_user_specified_namePlaylist_Inputs:]Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_nameAudio_Feature_Inputs


1__inference_Track_Recommender_layer_call_fn_40520
playlist_inputs
audio_feature_inputs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	@
	unknown_8:@
	unknown_9:	@

unknown_10:@

unknown_11:


unknown_12:	

unknown_13:


unknown_14:	

unknown_15:


unknown_16:	

unknown_17:


unknown_18:	

unknown_19:	

unknown_20:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallplaylist_inputsaudio_feature_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_40473o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)
_user_specified_namePlaylist_Inputs:]Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_nameAudio_Feature_Inputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
U
Audio_Feature_Inputs=
&serving_default_Audio_Feature_Inputs:0ÿÿÿÿÿÿÿÿÿ
O
Playlist_Inputs<
!serving_default_Playlist_Inputs:0ÿÿÿÿÿÿÿÿÿ
D
Final_Prediction0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Öì
Ê
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
»

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
»

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
»

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
»

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Gkernel
Hbias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
»

]kernel
^bias
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
»

ekernel
fbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
»

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
»

ukernel
vbias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
¢
}iter

~beta_1

beta_2

decay
learning_rate
momentum_cachemÕ mÖ'm×(mØ/mÙ0mÚ7mÛ8mÜ?mÝ@mÞGmßHmàUmáVmâ]mã^mäemåfmæmmçnmèumévmêvë vì'ví(vî/vï0vð7vñ8vò?vó@vôGvõHvöUv÷Vvø]vù^vúevûfvümvýnvþuvÿvv"
	optimizer
Æ
0
 1
'2
(3
/4
05
76
87
?8
@9
G10
H11
U12
V13
]14
^15
e16
f17
m18
n19
u20
v21"
trackable_list_wrapper
Æ
0
 1
'2
(3
/4
05
76
87
?8
@9
G10
H11
U12
V13
]14
^15
e16
f17
m18
n19
u20
v21"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
1__inference_Track_Recommender_layer_call_fn_40520
1__inference_Track_Recommender_layer_call_fn_41035
1__inference_Track_Recommender_layer_call_fn_41085
1__inference_Track_Recommender_layer_call_fn_40855À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þ2û
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_41171
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_41257
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_40917
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_40979À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
éBæ
 __inference__wrapped_model_40259Playlist_InputsAudio_Feature_Inputs"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
-
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ý2Ú
3__inference_Platlists_Flattened_layer_call_fn_41314¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_Platlists_Flattened_layer_call_and_return_conditional_losses_41320¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,:*
2Playlists_Dense_1/kernel
%:#2Playlists_Dense_1/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_Playlists_Dense_1_layer_call_fn_41329¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_Playlists_Dense_1_layer_call_and_return_conditional_losses_41340¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,:*
2Playlists_Dense_2/kernel
%:#2Playlists_Dense_2/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_Playlists_Dense_2_layer_call_fn_41349¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_Playlists_Dense_2_layer_call_and_return_conditional_losses_41360¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,:*
2Playlists_Dense_3/kernel
%:#2Playlists_Dense_3/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_Playlists_Dense_3_layer_call_fn_41369¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_Playlists_Dense_3_layer_call_and_return_conditional_losses_41380¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
(:&	2Tracks_Dense_1/kernel
": 2Tracks_Dense_1/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_Tracks_Dense_1_layer_call_fn_41389¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_Tracks_Dense_1_layer_call_and_return_conditional_losses_41400¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
*:(	@2Playlists_Latent/kernel
#:!@2Playlists_Latent/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_Playlists_Latent_layer_call_fn_41409¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_Playlists_Latent_layer_call_and_return_conditional_losses_41420¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
':%	@2Tracks_Latent/kernel
 :@2Tracks_Latent/bias
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_Tracks_Latent_layer_call_fn_41429¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_Tracks_Latent_layer_call_and_return_conditional_losses_41440¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
à2Ý
6__inference_Concatenated_Encodings_layer_call_fn_41446¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
û2ø
Q__inference_Concatenated_Encodings_layer_call_and_return_conditional_losses_41453¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
':%
2Prediction_1/kernel
 :2Prediction_1/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_Prediction_1_layer_call_fn_41462¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_Prediction_1_layer_call_and_return_conditional_losses_41473¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
':%
2Prediction_2/kernel
 :2Prediction_2/bias
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_Prediction_2_layer_call_fn_41482¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_Prediction_2_layer_call_and_return_conditional_losses_41493¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
':%
2Prediction_3/kernel
 :2Prediction_3/bias
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_Prediction_3_layer_call_fn_41502¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_Prediction_3_layer_call_and_return_conditional_losses_41513¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
':%
2Prediction_4/kernel
 :2Prediction_4/bias
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_Prediction_4_layer_call_fn_41522¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_Prediction_4_layer_call_and_return_conditional_losses_41533¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
*:(	2Final_Prediction/kernel
#:!2Final_Prediction/bias
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_Final_Prediction_layer_call_fn_41542¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_Final_Prediction_layer_call_and_return_conditional_losses_41553¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
 "
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
0
Ê0
Ë1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
æBã
#__inference_signature_wrapper_41309Audio_Feature_InputsPlaylist_Inputs"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

Ìtotal

Ícount
Î	variables
Ï	keras_api"
_tf_keras_metric
c

Ðtotal

Ñcount
Ò
_fn_kwargs
Ó	variables
Ô	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
Ì0
Í1"
trackable_list_wrapper
.
Î	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ð0
Ñ1"
trackable_list_wrapper
.
Ó	variables"
_generic_user_object
2:0
2 Nadam/Playlists_Dense_1/kernel/m
+:)2Nadam/Playlists_Dense_1/bias/m
2:0
2 Nadam/Playlists_Dense_2/kernel/m
+:)2Nadam/Playlists_Dense_2/bias/m
2:0
2 Nadam/Playlists_Dense_3/kernel/m
+:)2Nadam/Playlists_Dense_3/bias/m
.:,	2Nadam/Tracks_Dense_1/kernel/m
(:&2Nadam/Tracks_Dense_1/bias/m
0:.	@2Nadam/Playlists_Latent/kernel/m
):'@2Nadam/Playlists_Latent/bias/m
-:+	@2Nadam/Tracks_Latent/kernel/m
&:$@2Nadam/Tracks_Latent/bias/m
-:+
2Nadam/Prediction_1/kernel/m
&:$2Nadam/Prediction_1/bias/m
-:+
2Nadam/Prediction_2/kernel/m
&:$2Nadam/Prediction_2/bias/m
-:+
2Nadam/Prediction_3/kernel/m
&:$2Nadam/Prediction_3/bias/m
-:+
2Nadam/Prediction_4/kernel/m
&:$2Nadam/Prediction_4/bias/m
0:.	2Nadam/Final_Prediction/kernel/m
):'2Nadam/Final_Prediction/bias/m
2:0
2 Nadam/Playlists_Dense_1/kernel/v
+:)2Nadam/Playlists_Dense_1/bias/v
2:0
2 Nadam/Playlists_Dense_2/kernel/v
+:)2Nadam/Playlists_Dense_2/bias/v
2:0
2 Nadam/Playlists_Dense_3/kernel/v
+:)2Nadam/Playlists_Dense_3/bias/v
.:,	2Nadam/Tracks_Dense_1/kernel/v
(:&2Nadam/Tracks_Dense_1/bias/v
0:.	@2Nadam/Playlists_Latent/kernel/v
):'@2Nadam/Playlists_Latent/bias/v
-:+	@2Nadam/Tracks_Latent/kernel/v
&:$@2Nadam/Tracks_Latent/bias/v
-:+
2Nadam/Prediction_1/kernel/v
&:$2Nadam/Prediction_1/bias/v
-:+
2Nadam/Prediction_2/kernel/v
&:$2Nadam/Prediction_2/bias/v
-:+
2Nadam/Prediction_3/kernel/v
&:$2Nadam/Prediction_3/bias/v
-:+
2Nadam/Prediction_4/kernel/v
&:$2Nadam/Prediction_4/bias/v
0:.	2Nadam/Final_Prediction/kernel/v
):'2Nadam/Final_Prediction/bias/vÚ
Q__inference_Concatenated_Encodings_layer_call_and_return_conditional_losses_41453Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ@
"
inputs/1ÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ±
6__inference_Concatenated_Encodings_layer_call_fn_41446wZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ@
"
inputs/1ÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¬
K__inference_Final_Prediction_layer_call_and_return_conditional_losses_41553]uv0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_Final_Prediction_layer_call_fn_41542Puv0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¯
N__inference_Platlists_Flattened_layer_call_and_return_conditional_losses_41320]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ

ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
3__inference_Platlists_Flattened_layer_call_fn_41314P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ®
L__inference_Playlists_Dense_1_layer_call_and_return_conditional_losses_41340^ 0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_Playlists_Dense_1_layer_call_fn_41329Q 0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ®
L__inference_Playlists_Dense_2_layer_call_and_return_conditional_losses_41360^'(0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_Playlists_Dense_2_layer_call_fn_41349Q'(0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ®
L__inference_Playlists_Dense_3_layer_call_and_return_conditional_losses_41380^/00¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_Playlists_Dense_3_layer_call_fn_41369Q/00¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬
K__inference_Playlists_Latent_layer_call_and_return_conditional_losses_41420]?@0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
0__inference_Playlists_Latent_layer_call_fn_41409P?@0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@©
G__inference_Prediction_1_layer_call_and_return_conditional_losses_41473^UV0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_Prediction_1_layer_call_fn_41462QUV0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_Prediction_2_layer_call_and_return_conditional_losses_41493^]^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_Prediction_2_layer_call_fn_41482Q]^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_Prediction_3_layer_call_and_return_conditional_losses_41513^ef0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_Prediction_3_layer_call_fn_41502Qef0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_Prediction_4_layer_call_and_return_conditional_losses_41533^mn0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_Prediction_4_layer_call_fn_41522Qmn0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_40917º '(78/0?@GHUV]^efmnuvy¢v
o¢l
b_
-*
Playlist_Inputsÿÿÿÿÿÿÿÿÿ

.+
Audio_Feature_Inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_40979º '(78/0?@GHUV]^efmnuvy¢v
o¢l
b_
-*
Playlist_Inputsÿÿÿÿÿÿÿÿÿ

.+
Audio_Feature_Inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ø
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_41171§ '(78/0?@GHUV]^efmnuvf¢c
\¢Y
OL
&#
inputs/0ÿÿÿÿÿÿÿÿÿ

"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ø
L__inference_Track_Recommender_layer_call_and_return_conditional_losses_41257§ '(78/0?@GHUV]^efmnuvf¢c
\¢Y
OL
&#
inputs/0ÿÿÿÿÿÿÿÿÿ

"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ã
1__inference_Track_Recommender_layer_call_fn_40520­ '(78/0?@GHUV]^efmnuvy¢v
o¢l
b_
-*
Playlist_Inputsÿÿÿÿÿÿÿÿÿ

.+
Audio_Feature_Inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿã
1__inference_Track_Recommender_layer_call_fn_40855­ '(78/0?@GHUV]^efmnuvy¢v
o¢l
b_
-*
Playlist_Inputsÿÿÿÿÿÿÿÿÿ

.+
Audio_Feature_Inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÐ
1__inference_Track_Recommender_layer_call_fn_41035 '(78/0?@GHUV]^efmnuvf¢c
\¢Y
OL
&#
inputs/0ÿÿÿÿÿÿÿÿÿ

"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÐ
1__inference_Track_Recommender_layer_call_fn_41085 '(78/0?@GHUV]^efmnuvf¢c
\¢Y
OL
&#
inputs/0ÿÿÿÿÿÿÿÿÿ

"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿª
I__inference_Tracks_Dense_1_layer_call_and_return_conditional_losses_41400]78/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_Tracks_Dense_1_layer_call_fn_41389P78/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
H__inference_Tracks_Latent_layer_call_and_return_conditional_losses_41440]GH0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
-__inference_Tracks_Latent_layer_call_fn_41429PGH0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@õ
 __inference__wrapped_model_40259Ð '(78/0?@GHUV]^efmnuvq¢n
g¢d
b_
-*
Playlist_Inputsÿÿÿÿÿÿÿÿÿ

.+
Audio_Feature_Inputsÿÿÿÿÿÿÿÿÿ
ª "Cª@
>
Final_Prediction*'
Final_Predictionÿÿÿÿÿÿÿÿÿ¢
#__inference_signature_wrapper_41309ú '(78/0?@GHUV]^efmnuv¢
¢ 
ª
F
Audio_Feature_Inputs.+
Audio_Feature_Inputsÿÿÿÿÿÿÿÿÿ
@
Playlist_Inputs-*
Playlist_Inputsÿÿÿÿÿÿÿÿÿ
"Cª@
>
Final_Prediction*'
Final_Predictionÿÿÿÿÿÿÿÿÿ