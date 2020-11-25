{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving  #-}
{-# LANGUAGE TypeOperators       #-}

module BPTT where 

import Control.Monad
import Control.Monad.Random
import Data.Char
import Data.List
import Data.Maybe
import Data.Singletons
import Data.Singletons.Prelude
import Data.Singletons.TypeLits
import GHC.Show
import Numeric (showIntAtBase)
import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra.Static

-- dependently typed weights
data Weights i o = W {
    wBiases :: !(R o) 
  , wNodes  :: !(L o i)
  }

-- an simple 3-layer RNN, including one recurrent layer,
-- is not like the Network type discussed in Justin Le's 
-- blog series (because not all layers are just stacked
-- one on top of the other.)
-- Instead:
data RNN i h o = MkRNN {
    getU    :: !(Weights i h)
  , getW    :: !(Weights h h)  
  , getV    :: !(Weights h o)
  }

-- If one were to leverage a.g. ad the Hackage library for 
-- autodifferentiable functions, then functions being used for
-- activation could be provided in a symbolic form amenable to
-- being automatically differentiated (for use in the backprop
-- process.) Instead, each activation function used & its derivative
-- are provided separately.

logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))

logistic' :: Floating a => a -> a
logistic' x = logix * (1 - logix)
  where
    logix = logistic x

softmax :: KnownNat v => R v -> R v 
softmax v = 
  let vexp = exp v 
  in  vexp / (konst $ norm_1 vexp)

-- The formula for the (x, y)-entry of softmax',
-- which is a matrix, is 
--
-- softmax'[x, y] = -softmax[x]    * softmax[y],    x /= y
--                  (1-softmax[x]) * softmax[y],    x == y
--
-- which is the same as adding the vector 
-- 'softmax v' diagonally down the negative of 
-- said vector's outer product with itself.
-- I would prefer to build this using 'build', but 
-- I don't know how right now.
softmax' :: KnownNat v => R v -> L v v 
softmax' v = 
  let s = softmax v 
  in  (diag s) + konst (-1) * (s `outer` s)

-- runLayer is for performing the linear mapping
-- due to a particular layer's set of parameters,
-- on an appropriately sized vector
runLayer :: (KnownNat i, KnownNat o) 
         => Weights i o 
         -> R i -> R o 
runLayer (W wB wN) v_in = 
  wB + wN #> v_in

  

-- -- GIVEN: rnn                    :: RNN i h o
-- --        current input vector   :: R i 
-- --        prior hidden state     :: R h
-- -- GIVES: current prediction     :: R o
-- --        *and* new hidden state :: R h
-- runRNN :: (KnownNat i, KnownNat h, KnownNat o)
--        => RNN i h o
--        -> R i -> R h 
--        -> (R o, R h)
-- runRNN (MkRNN rnn_U rnn_W rnn_V) 
--   x_t h_t_1 = 
--     let h_t = logistic $ 
--                 (runLayer rnn_W h_t_1) + 
--                 (runLayer rnn_U x_t)
--         z_t = runLayer rnn_V h_t 
--         y_t = logistic z_t 
--     in  (y_t, h_t)

-- Verbose 'run RNN', for exposing all intermediate computations.
-- (R i, R h,  R h, R o, R o)
--  x_t  h_t-1 h_t  z_t  y_t    

-- runRNN can be expressed in terms of this, if I locate / find 
-- generalized tuple accessors
runRNN_verbose  :: (KnownNat i, KnownNat h, KnownNat o)
                => RNN i h o
                -> R i -> R h 
                -> (R i, R h, R h, R o, R o)
runRNN_verbose (MkRNN rnn_U rnn_W rnn_V) 
  x_t h_t_1 = 
    let h_t = logistic $ 
                -- recurrent parameters, weighting the prior hidden state
                (runLayer rnn_W h_t_1) + 
                -- input parameters, weighting the current input
                (runLayer rnn_U x_t)
        -- output parameters, weighting the current hidden state
        z_t = runLayer rnn_V h_t 
        -- the final activation provides the prediction 
        y_t = logistic z_t 
    in  (x_t, h_t_1, h_t, z_t, y_t)  

-- Forward phase of training an RNN.
-- No adjustments to the parameters (weights) are made.
-- Instead, foldl is used to accumulate a list of tuples,
-- each tuple bearing all input, intermediate and output vectors
-- for one of the steps (t := 1, 2, ..., n) in the forward direction.
forward_phase :: (KnownNat i, KnownNat h, KnownNat o)
              => RNN i h o    -- RNN
              -> [(R i, R o)] -- list of training inputs paired with target outputs
              -> R h          -- initial hidden state

                              -- All tuples of resulting vectors, from every step carried out:
                              -- For t ranging from 1 to n, n the number of training samples:
              -> [( R i,      -- x_t
                    R h,      -- h_t-1
                    R h,      -- h_t
                    R o,      -- z_t
                    R o,      -- y_t
                    R o)]     -- tgt_t
forward_phase 
  rnn 
  pairs 
  h_0 = 
    snd $ foldl unroll (h_0, []) pairs
    where 
      -- unroll :: (R h, [(R i, R h, R h, R o, R o, R o)]) -> (R i, R o)
      --        -> (R h, [(R i, R h, R h, R o, R o, R o)])

      unroll (h_t_1, results) pair@(x_t, tgt_t) = 
        --                               ^ tgt_t plays no role in the forward direction,
        --                                 but we pattern-match on it so that we can re-package it
        --                                 up into this step's tuple for the result list
        let result@(_, _, h_t, z_t, y_t) = runRNN_verbose rnn x_t h_t_1 
        in  (h_t, results ++ [(x_t, h_t_1, h_t, z_t, y_t, tgt_t)]) 

-- Backward phase of training an RNN.
-- Backpropagation is required.
-- There's a lot that goes into backprop!
-- (Basically, everything that is used or 
-- produced at any point in the forward step.)
backward_phase  :: (KnownNat i, KnownNat h, KnownNat o)
                => RNN i h o   -- an RNN to train

                                -- All tuples of resulting vectors, from every step carried out:
                                -- For t ranging from 1 to n, n the number of training samples:
                -> [( R i,      -- x_t
                      R h,      -- h_t-1
                      R h,      -- h_t
                      R o,      -- z_t
                      R o,      -- y_t
                      R o)]     -- tgt_t

                -- At every step #t of BPTT, there are *two* sources of "delta-h_t" the bump
                -- to h_t required for the sake of trying to bump down the error at that same step:
                -- a) a bump due to what W is at the current time, "delta-h given W", and
                -- b) a bump *propagated backwards* from the next step #(t+1).
                -- But in the base case, this dh_t_dh_t' is chosen as 0 (because it's undefined.)

                -- Note the *symmetry* with forward_phase:
                --    just as we 'primed' forward_phase  with an initial hidden state,    h_0        : 0,
                --    we also    'prime'  backward_phase with an initial back-prop'ed dh, dh_n_dh_n' : 0.
                -> R h        -- dh_n_dh_n'

                -> Double     -- a training rate
                -> RNN i h o  -- the trained RNN
backward_phase rnn intermediates dh_n_dh_n' rate =
  snd $ foldr bptt (dh_n_dh_n', rnn) intermediates  
    where 
      -- -- A BPTT step (at time t) consumes the following data:
      -- --       x_t, h_t, h_t_1, z_t,  y_t, tgt_t
      -- bptt :: (R i, R h, R h,   R o,  R o, R o)
      -- -- It also "accumulates" by swapping out,
      -- -- based upon time-t data  <dh_t   | dh_t+1>, rnn
      --          -> (R h, RNN i h o)
      -- -- the time-(t-1) data     <dh_t_1 | dh_t>,   rnn';
      -- -- in this way, the rnn' "accumulates" training bumps to the parameters
      --          -> (R h, RNN i h o)
      
      bptt    (x_t, h_t, h_t_1, z_t, y_t, tgt_t) 
              (dh_t_dh_t', MkRNN 
                -- We'll pattern match into each Weights term u, w, v for convenience 
                -- later in deriving the desired infinitesimal bumps.
                u@(W uB uN) 
                w@(W wB wN) 
                v@(W vB vN)) = 
                -- We are hard-coding the use of squared-error for the error function:
                -- e(y_t) := (y_t - tgt_t) ^ 2
                -- whose derivative with respect to the pre-activation z_t is:

                {-
                  de/dz = de/dy           * dy/dz
                        = (e' at y_t)     * (y' at z_t)
                        = 2 (y_t - tgt_t) * (logistic' z_t)
                  These 2 factors appear (in reverse order) as 'dedz' below;
                  the factor of 2 is discarded & we defer all control of the 
                  desired training rate to the scalar input param 'rate'.
                -}
            let dedz          = logistic' z_t * (y_t - tgt_t)
                              -- = softmax' z_t #> (y_t - tgt_t)

                {-
                  Informally:
                  dedz is a vector that reports, 
                    for a unit "nudge" to each component z_t(j) -> 
                      size / direction of resulting "nudge" for error component e_t(j).
                  The choice of dz_t below corresponds to the following training strategy:
                    a) bump each z-component in the direction that will *decrease* its 
                       matching e-component (thus, factor of '-1')
                    b) but re-scaled based on the learning rate desired (thus, 'scale rate')
                -}
                dz_t          = -1 * konst rate * dedz 
                
                {-
                  Informally:
                  Having set out to make these infinitesimal changes to the z_t components,
                  we need to figure out 
                    what changes to V,    *given* what h_t is,     and
                    what changes to h_t,  *given* what V is
                  will help us accomplish that.

                  Observe that 
                    z  = vN h              + vB     
                  therefore (product rule on 1st summand)
                    dz = (dvN h) + (vN dh) + dvB

                  we do *not* try to formally solve for dvN, dh, dvB in here, but rather use these
                  as instructions for how to recombine the vectors dz_t / h_t and matrix V
                  so as to arrive at dvN_h ("bump to v's matrix part, given h") and 
                  dh_vN ("bump to h, given v's matrix part").

                  As for v's bias-vector part, vB, we simply nudge it directly by the dz (z-nudge)
                  we worked out above. Why? Look at the infinitesimals equation above - a nudge 
                  dvB performed on the bias vector will show up directly in dz.
                -}
                dvB           = dz_t
                {-
                  Instead of trying to solve for dvN, dimensional considerations lead us to this formula
                  for dvN in terms of dz_t, h_t:
                    vN   goes    from h-vectors  to o-vectors; i.e.    o * h -matrix
                    h_t            an h-vector
                    dz_t                         an o-vector
                    so the way to combine these to get a "bumps matrix" compatible with vN is
                    dz_t `outer` h_t   <- "column" o-vector times "row" h-vector
                -}
                dvN_h         = dz_t `outer` h_t 
                {-
                  Similar to the above, but for building a "bumps h-vector" out of o * h matrix 
                  and o-vector.
                -}
                dh_vN         = tr vN #> dz_t 
                {-
                  See the note in the type signature of 'backward_phase' about the 2 sources of 
                  training bumps to h_t at every step t.
                  Read dh_t_dh_t' as (bump to h at current time t,
                                      propped backwards from bump to h at future time t+1)
                -}
                dh_t          = dh_vN + dh_t_dh_t' 

                -- Having now characterized in what way we are going to bump the 
                -- entries in the current hidden state h_t, so as to achieve
                -- a desired reduction in error (i.e. dh_t), we now use it to
                -- determine how we should modify the intermediate preactivation,
                -- calling it q, q_t := (W * h_t_1 + U * x_t) to achieve the same goal:

                -- de / dq = (de / dh) * (dh / dq)  -- chain rule
                --         = dh_t      * (dh / dq)  -- back propagation
                --         = dh_t      * (g' at q)  -- h is the activation (g) of q

                dedq          = dh_t * logistic' q 
                                  where q = runLayer w h_t_1 + runLayer u x_t 
                
                -- and work out a desired perturbation to q, based upon scaling dedq                   
                dq_t          = (-1) * konst rate * dedq 

                {-
                  q  = W * h_t_1 + U * x_t
                  dq = (dW * h_t_1) + (W * dh_t_1) 
                        + (dU * x_t) + [the term in dx_t -> 0, since x constant]
                -}

                {-
                  Recurrent contribution to time-t hidden state:
                  q_t  = wN h_t_1                   + wB        + ...
                  dq_t = (dwN h_t_1) + (wN dh_t_1)  + dwB       + ...
                -}
                dwB           = dq_t
                -- dq_t and h_t_1 both happen to be h-vectors having the same
                -- dimensionality, so here the driving intuition is that such
                -- "matrix-bumps" take the form [destination] `outer` [origin]
                dwN_h         = dq_t `outer` h_t_1

                -- This is what will be given back from the current fold instance
                -- to the next one 
                dh_t_1_dh_t   = tr wN #> dq_t 
                {-
                  Input contribution to time-t hidden state:
                  q_t  = ... + uN x_t                 + uB 
                  dq_t = ... + (duN x_t) + (uN dx_t)  + duB
                       = ... + (duN x_t) + ZERO       + duB
                                        -- ^ x_t, the training inputs, are constant!
                -}
                duB           = dq_t 
                duN_x         = dq_t `outer` x_t


                -- Don't forget to apply all the RNN-training nudges!
                vB'           = vB + dvB
                vN'           = vN + dvN_h
                wB'           = wB + dwB
                wN'           = wN + dwN_h
                uB'           = uB + duB
                uN'           = uN + duN_x

            in  (dh_t_1_dh_t, MkRNN (W uB' uN') 
                                    (W wB' wN') 
                                    (W vB' vN')) 


-- IDEAS:
-- 1. NEWTYPES, for working "mathematical 
--    variable naming conventions" into the 
--    code: aids human-readability
-- 2. DEPENDENT TYPES, so that the types 
--    of the training data / intermediate results
--    / network parameters being passed around 
--    carry information about their dimensions

{-
 Making some simple training data. 

 Here's a simple sentential grammar for producing "sentences"
 of ints, that later can go in an overall "document":

  <sentence>        ::= <sentence-start> <sentence-middle> <sentence-finish>
  <sentence-start>  ::= a multiple of 20, 20 * s, for some s : 1 <= s <= 10
  <sentence-middle> ::= a sequence of l integers, for some l : 0 <= l <= 10
                        that are derived from the sentence-start as 
                          s - m, (s - m) + n, (s - m + n) - m, (s - m + n - m) + n, ...
                        for some choice of parameters      m, n : 1 <= m, n <= 5
  <sentence-finish> ::= a multiple of 17, 17 * f, for some f : 1 <= f <= 10
-}

-- Generate a "biarithmetic" sequence for sentence-middle, given 
-- a base number s, a down size m, an up size n 
biarithmetic :: Int -> Int -> Int -> [Int]
biarithmetic    s      m      n   =  concat $ 
 [ [ s - (j * m) + (k * n) | k <- [j-1 .. j] ] 
                           | j <- [1..] ]

-- For use as a multiple of 20
genStartParam :: MonadRandom m => m Int 
genStartParam = getRandomR (3, 10)

-- For use as a mid-sequence length 
genMiddleLenParam :: MonadRandom m => m Int 
genMiddleLenParam = getRandomR (0, 10)

-- For use as the down size and up size of a biarithmetic sequence
genMiddleDownUpParams :: MonadRandom m => m (Int, Int)
genMiddleDownUpParams = do 
  m <- getRandomR (1, 5)
  n <- getRandomR (1, 5)
  return (m, n)

-- For use as a multiple of 17
genFinishParam :: MonadRandom m => m Int 
genFinishParam = getRandomR (1, 10)

-- For use as a number of "sentences".
-- Bump this and / or genMiddleLenParam up once I'm confident the BPTT terminates 
-- on smaller sizes.
genSentenceCount :: MonadRandom m => m Int 
genSentenceCount = getRandomR (500, 1000)

type Sentence = [Int]

genSentence :: MonadRandom m => m Sentence 
genSentence = do 
  -- 1. Get sentence-start
  s       <- genStartParam
  let sent_0 = s * 20 
  -- 2. Get sentence-middle length
  l       <- genMiddleLenParam
  -- 3. Get sentence up & down size params
  (m, n)  <- genMiddleDownUpParams
  -- 4. Get sentence-middle
  let sent_mid = take l $ biarithmetic sent_0 m n 
  -- 5. Get sentence-finish
  f       <- genFinishParam
  let sent_f = f * 17
  -- 6. Return random sentence
  return (sent_0 : sent_mid ++ [sent_f])

type Document = [Sentence]

genDocument :: MonadRandom m => m Document 
genDocument = do
  -- 1. Get number of sentences
  sent_count <- genSentenceCount
  -- 2. For as many sentences as 'sent_count', 
  let rand_sent_stream = repeat genSentence
  sentences <- sequenceA $ take sent_count rand_sent_stream
  return sentences 

{-
 Now that we have a way of generating "documents" made up out of 
  numerical "sentences", we need to convert this into a form that
  would be characteristically used to train a neural net.
  Convert each integer to a (possibly sparse) vector. How? 
  Every integer present in a (d :: Document) should always lie
  within the range (0 <= j <= 256), so we could use a binary representation
  taking up 8 bits (or tossing in an extra to get 9, for good measure)
-}

-- The full source module citation is probably forgivable if I only do it this once.
-- Best practice is to 'import qualified' with an alias ...
intToVector :: Int -> R 10 
intToVector m = 
  let binary_string = reverse $ showIntAtBase 2 intToDigit m "" -- :: [Char]
      binary_padded = binary_string ++ 
                        (take (10 - length binary_string) $ repeat '0')
      doubles_list  = (fromIntegral . digitToInt) <$> binary_padded
      doubles_vect  = Numeric.LinearAlgebra.Static.fromList doubles_list
  in  doubles_vect

-- The other way round, for casting RNN predictions back into Int's.
-- This is unsafe in case the entries of any y_t end up rounding to
-- numbers other than 0, 1.
-- (Does the hmatrix API support a more direct way of casting from
-- R 10 to the list of rounded numbers?)
vectorToInt :: R 10 -> Int 
vectorToInt v = 
  let bins = (fmap toInteger) <$> LA.toList 
                                $ LA.toZ 
                                $ LA.roundVector 
                                $ extract v 
  in  fromIntegral $ 
        foldr (\p@(bit,  power)  total -> 
                   bit * power + total) 
        0
        (zip bins [ 2^j | j <- [0..] ])


-- TODO:  1. function to create zeroes vector of a given input dimension
--        2. random generation of Weights
--        3. random generation of RNN
--        4. overall training function that takes a random RNN & Document,
--           then trains
--        5. then, testing function that takes a trained RNN & Document, and
--           assesses performance
--           *or* function that takes a trained RNN & a suitable initial integer,
--           then unfolds a sentences based upon series of predictions
--        6. also put softmax / softmax' in here?

zero_vec :: KnownNat n => R n
zero_vec = 0

-- via Justin Le's blog series on neural nets
randomWeights :: (MonadRandom m, KnownNat i, KnownNat o)
              => m (Weights i o)
randomWeights = do
    s1 :: Int <- getRandom
    s2 :: Int <- getRandom
    let wB = randomVector  s1 Uniform * 2 - 1
        wN = uniformSample s2 (-1) 1
    return $ W wB wN

randomRNN :: (MonadRandom m, KnownNat i, KnownNat h, KnownNat o)
          => m (RNN i h o)
randomRNN = 
  (pure MkRNN)  <*> randomWeights
                <*> randomWeights
                <*> randomWeights

trainRNN :: (KnownNat i, KnownNat h, KnownNat o) 
         => RNN i h o     -- an RNN 
         -> [(R i, R o)]  -- the training data: pairs of inputs & training outputs
         -> Double 
         -> RNN i h o     -- an RNN with trained parameters
trainRNN rnn train_data rate = 
  let intermediates = forward_phase rnn train_data zero_vec 
      trained       = backward_phase rnn intermediates zero_vec rate
  in  trained

-- Before re-formatting into a vector form suitable for use 
-- in training, we have Document ~ [Sentence] ~ [[Int]];
-- now, turn each Int into a binary-repr vector, R 10
genSampleData :: MonadRandom m => m [[ R 10 ]]
genSampleData = do 
  doc <- genDocument 
  return $ (intToVector <$>) <$> doc

-- Every sentence (sublist) in the document (list of lists) will
-- furnish one sequence of training data for the RNN to "unfold"
-- over and perform BPTT on. So, BPTT is performed once per sentence
-- in the document overall.
pairEachWithNext :: [x] -> [(x, x)]
pairEachWithNext xs = 
  zip (init xs) (tail xs)

-- In the special case of an 'RNN i h o' where o = i,
-- we can pipe back a prediction y_t due to x_t and an
-- initial state h, to furnish the next x_(t+1) for further
-- predictions. We'll use this to observe the way that a
-- trained RNN behaves.
rnnStep :: (KnownNat i, KnownNat h) 
        => RNN i h i
        -> (R h, R i)
        -> (R h, R i)
rnnStep rnn (h_t_1, x_t) = 
  let full_output = runRNN_verbose rnn x_t h_t_1 
      third five@(_, _, x3, _, _)  = x3 
      fifth five@(_, _, _,  _, x5) = x5 
  in  (third full_output, fifth full_output)

-- Randomly generate an RNN & some document data to train
-- it on, performing the training & giving back the trained
-- RNN within IO.

getRnnIO :: IO (RNN 10 300 10)
getRnnIO = evalRandIO randomRNN 

{-
 e.g.

 getRnnIO >>= 
  (\rnn -> return $ vectorToInt <$> takePredsFromN rnn 20 5)
-}

getDocumentIO :: IO Document 
getDocumentIO = evalRandIO genDocument

takePredictions  :: (KnownNat i, KnownNat h) 
                 => RNN i h i
                 -> (R h, R i)
                 -> Int 
                 -> [R i]
takePredictions rnn (h_0, x_1) n = 
  let hxs_stream = take (n+1) $ iterate (rnnStep rnn) (h_0, x_1)
  in  snd <$> hxs_stream

takePredsFromN :: KnownNat h
               => RNN 10 h 10
               -> Int      -- word of a sentence
               -> Int      -- number of predictions
               -> [R 10]
takePredsFromN rnn x_1 n = 
  takePredictions rnn (zero_vec, intToVector x_1) n

f x_1 n = 
  pure takePredictions  <*> getRnnIO 
                        <*> pure (zero_vec, intToVector x_1)
                        <*> pure n 

g x_1 n =
  (vectorToInt <$>) <$> f x_1 n

trainRNNOnSentence  :: KnownNat h
                    => RNN 10 h 10 
                    -> Sentence
                    -> Double 
                    -> RNN 10 h 10 
trainRNNOnSentence rnn sent rate = 
  let training_data = pairEachWithNext (intToVector <$> sent)
  in  trainRNN rnn training_data rate 

-- trainRNNOnDocument has to be structured like a fold
-- so that the trained RNN due to each sentence can be 
-- passed to the following sentence (at the same time,
-- I think that hidden state "restarts" with the beginning
-- of a new sentence.)
trainRNNOnDocument  :: KnownNat h
                    => RNN 10 h 10 
                    -> Document 
                    -> Double 
                    -> RNN 10 h 10 
trainRNNOnDocument rnn sents rate = 
  foldl (\rnn sent -> trainRNNOnSentence rnn sent rate)
        rnn 
        sents

h = pure trainRNNOnDocument <*> getRnnIO
                            <*> getDocumentIO
                            <*> pure 0.20
{-
  e.g.
  h >>= 
    (\rnn -> return $ vectorToInt <$> takePredsFromN rnn 20 50)
-}
