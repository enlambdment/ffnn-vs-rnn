{-# LANGUAGE BangPatterns               #-}
{-# LANGUAGE ScopedTypeVariables        #-}
{-# LANGUAGE GADTs                      #-}
{-# LANGUAGE KindSignatures             #-}

module Main where 

import Control.Monad
import Control.Monad.Random
import Data.List
import Data.Maybe
import Numeric.LinearAlgebra
import System.Environment
import Text.Read


{-
 Weights, Network & neural net implementation by Justin Le (not mine!)

 Tutorial by Justin Le:

 https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html
-}
data Weights = W { wBiases :: !(Vector Double)  -- n
                 , wNodes  :: !(Matrix Double)  -- n x m
                 }                              -- "m to n" layer
 deriving Show 

data Network :: * where
    O     :: !Weights
          -> Network
    (:&~) :: !Weights
          -> !Network
          -> Network
  deriving Show -- ?

infixr 5 :&~

logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))

logistic' :: Floating a => a -> a
logistic' x = logix * (1 - logix)
  where
    logix = logistic x

runLayer :: Weights -> Vector Double -> Vector Double
runLayer (W wB wN) v = wB + wN #> v

runNet :: Network -> Vector Double -> Vector Double
runNet (O w)      !v = logistic (runLayer w v)
runNet (w :&~ n') !v = let v' = logistic (runLayer w v)
                       in  runNet n' v'

randomWeights :: MonadRandom m => Int -> Int -> m Weights
randomWeights i o = do
    seed1 :: Int <- getRandom
    seed2 :: Int <- getRandom
    let wB = randomVector  seed1 Uniform o * 2 - 1
        wN = uniformSample seed2 o (replicate i (-1, 1))
    return $ W wB wN

randomNet :: MonadRandom m => Int -> [Int] -> Int -> m Network
randomNet i []     o =     O <$> randomWeights i o
randomNet i (h:hs) o = (:&~) <$> randomWeights i h <*> randomNet h hs o

train :: Double           -- ^ learning rate
      -> Vector Double    -- ^ input vector
      -> Vector Double    -- ^ target vector
      -> Network          -- ^ network to train
      -> Network
train rate x0 target = fst . go x0
  where
    go :: Vector Double    -- ^ input vector
       -> Network          -- ^ network to train
       -> (Network, Vector Double)
    -- handle the output layer
    go !x (O w@(W wB wN))
        = let y    = runLayer w x
              o    = logistic y
              -- the gradient (how much y affects the error)
              --   (logistic' is the derivative of logistic)
              -- Here, E := (o - target) ^ 2 is implied
              dEdy = logistic' y * (o - target)
              -- new bias weights and node weights
              wB'  = wB - scale rate dEdy
              wN'  = wN - scale rate (dEdy `outer` x)
              w'   = W wB' wN'
              -- bundle of derivatives for next step
              dWs  = tr wN #> dEdy 
          in  (O w', dWs)
    -- handle the inner layers
    go !x (w@(W wB wN) :&~ n)
        = let y          = runLayer w x
              o          = logistic y
              -- get dWs', bundle of derivatives from rest of the net
              (n', dWs') = go o n
              -- the gradient (how much y affects the error)
              dEdy       = logistic' y * dWs'
              -- new bias weights and node weights
              wB'  = wB - scale rate dEdy
              wN'  = wN - scale rate (dEdy `outer` x)
              w'   = W wB' wN'
              -- bundle of derivatives for next step
              dWs  = tr wN #> dEdy
          in  (w' :&~ n', dWs)

{-
 Inserting my own training exercise here.
 This is inspired by the language-model task of predicting the following 
 element in a 4-gram, based upon the last three elements available.
 The task: train a randomly generated system of weights to recognize 
 the following pattern & predict the following item according to it:

 0 1 2 0 1 2 0 1 2 ..
-}

inf_pattern :: [Int]
inf_pattern = cycle [0..2]

fin_pttrn_len :: Int
fin_pttrn_len = 99999

{-
 Forming training data from the inf_pattern.
 The training data will be a list of pairs, each pair containing 
 * a training input, 3-element vector;
 * a target output, 1-element vector.
-}
training_data :: [(Vector Double,  -- 3 * 1
                   Vector Double)] -- 1 * 1
training_data = 
  let fin_pattern = take fin_pttrn_len inf_pattern 
      list_pairs =
        [ (take 3 sub_pattern, [sub_pattern !! 3]) 
            | j <- [0..(length fin_pattern - 4)],
              let sub_pattern = drop j fin_pattern ]
  in map (\(inp, out) -> (vector $ fromIntegral <$> inp, 
                          vector $ fromIntegral <$> out))
         list_pairs

-- Actually, if we were really being realistic about this then we would
-- make our inputs and outputs work according to some sort of one-hot encoding.
-- But never mind.

-- 1. Get an initial random network. We'll pick a basic enough size:
myRandNet :: MonadRandom m => m Network 
myRandNet = randomNet 3 [4, 5] 1 

-- 2. How can I get a pre-training prediction, then iteratively
--    train on the next available sample, followed by getting the
--    revised prediction?

-- An auxiliary function for "sequencing together" 
-- training a network & printing info about it.
foldTraining :: Double              -- training rate
             -> Network             -- untrained network
             -> [(Vector Double,    -- pairs of training input
                  Vector Double)]   --          and training target output
             -> IO Network          -- train steps & print steps alternate in IO 
foldTraining rate net pairs = 
  foldl (\io_net sample@(inp, tgt) -> do 
            n <- io_net
            putStrLn $ "Will train on: " ++ show inp 
            let prediction = runNet n inp
            putStrLn $ "Predicted following item: " ++ show prediction
            -- now trying 10 rounds each per data sample
            let n' = last $ 
                     take 10 $ 
                     iterate (train rate inp tgt) n 
                          -- ^ :: Network -> Network
            return n')
        (return net)
        pairs

-- We can come back later & modify this to give back the resulting
-- trained network, for use elsewhere (i.e. recurrent NN's!)
displayTraining :: Double              -- training rate
                -> [(Vector Double,    -- pairs of training input
                     Vector Double)]   --          and training target output
                -> IO ()
displayTraining rate pairs = do 
  -- pluck out a random net 
  randNet <- evalRandIO myRandNet 
  -- show what it predicts for 1st sample *before* any training is done
  let inp_0 = fst $ head training_data
  let tgt_0 = snd $ head training_data 
  let predict_0 = runNet randNet inp_0
  putStrLn $ "The untrained network predicts that following: "
             ++ show inp_0 ++ ", we will get: " ++ show predict_0
  -- now, alternate showing predictions & performing training on some input
  foldTraining rate randNet pairs
  return ()

{-
 Remark.
 The network is obviously learning *something*, because early predictions
 (which are basically noise) eventually give way to predicting minuscule 
 numbers to follow every instance of "2", and numbers infinitesimally 
 less than "1" following every instance of "0". But learning that every 
 instance of "1" is followed by a "2" seems to take much longer! I haven't
 witnessed it yet despite using up to 10K training sample, with 1 learning
 round per sample.

 What if I try more rounds? It doesn't seem to help as far as training the 
  network to detect the "2 is followed by 0" part of the pattern.
-}

{-
 RECURRENT NEURAL NETWORKS.
 We study a simple case of these, Elman network a.k.a. simple RNN.
 * for ea. training sample, we simply take *one element* of the training
   data at a particular point in the sequence (rather than the window-base
   n-gram approach of before.)
 * for the training step at current time t, we rely upon:
    a) the training input at current time t, AND
    b) the activation from previous time (t-1)
        (so we have to initialize a "blank activation" for use 
         at the very first step, t := 1; this is h := the zero vector.)
    c) therefore, in a simple model with just *two* sets of weights
       (besides the one which handles the previous activation), the 
       back-propagation due to comparing current output vs. current target
       will back-propagate a bundle of "nudges" dh_t given what W2_t 
       is at the current time; 
       this in turn must back-prop *weight deltas* to 
       BOTH U_t (which applies to the previous activation)
       AND W1_t (which applies to the current training input.)
-}

-- RNN ~ Weights x Network.
-- We will model this using a separate collection of weights,
-- U, that we simply carry along while remembering to train.
data RNN = MkRNN {
  wts :: Weights,   -- <- U: h-to-h
  net :: Network }  -- <- (W: i-to-h) :&~ (V: h-to-o)

runRNN :: RNN 
       -> Vector Double -- input vector
       -> Vector Double -- previous-step activation
       -> Vector Double -- prediction
runRNN rnn@(MkRNN 
  u@(W uB uN) 
  n@(
    w@(W wB wN) :&~
    O v@(W vB vN))) x_t h_t_1 = 
    let z_u   = (uB + uN #> h_t_1) + -- weighted previous activation
                (wB + wN #> x_t)     -- weighted current training input
        h_t   = logistic z_u         -- current activation.
                                     -- *Need to include somewhere in output!*
        z_h   = (vB + vN #> h_t)     -- next layer of weights
        y_t   = logistic z_h         -- current final activation i.e. output
    in  y_t  -- ^ softmax usually, for tagging problems & similar 
             --   (multiple way classification)

{-
 The below is *not* actually a full implementation of the simple RNN
 described at the beginning of Jurafsky Ch. 9, because it does not
 perform the overall training in the two-pass algorithm which is
 explained there.
 In particular, the first pass is more or less all here, but the 
 second pass (Backpropgataion Through Time) is *not* taking place right now.
 The idea is actually to *wait* (I think?) until all of y_1, ..., y_l have been
 attempted, *then* (in phase #2) work backwards & successive apply gradient
 terms to V, U, W, U, W, U, W ... (? sequence of matrices being adjusted),
 propagating error back through h_l, h_(l-1), h_(l-2) ... , h_1 *all in one phase*!

 The idea is that, instead of just nudging the matrix entries of U (the recurrent
 part) once per training sample, you would actually wait until the end, and *then* 
 have steps iteratively unfolding along the lines of:

 dh_t       = tr vN #> dEdz
 dh_(t-1)   = tr uN #> dEdh_t
 dh_(t-2)   = tr uN #> dEdh_(t-1)
 dh_(t-3)   = tr uN #> dEdh_(t-2)
 ...

-}

trainRNN :: RNN -- ~ U x (W :&~ O V) :
                -- the  graph for this neural network is
                -- as follows:
                {-
                  h_t-1 -> U -> h_t -> V -> y_t
                              ^
                  x     -> W /
                -}
          -> Double        -- training rate
          -> Vector Double -- activation, h_t-1
          -> Vector Double -- training input, x_t
          -> Vector Double -- training target output, tgt_t
          -> (Vector Double, RNN) -- the new activation, h_t, and 
                                  -- all three weight-sets of the RNN trained
-- -- don't write recursions, just do everything by hand
trainRNN rnn@(MkRNN 
  u@(W uB uN) 
  n@(
    w@(W wB wN) :&~
    O v@(W vB vN))) rate h_t_1 x_t tgt_t = 
  -- compute the model output
  let z_u   = (uB + uN #> h_t_1) + -- weighted previous activation
              (wB + wN #> x_t)     -- weighted current training input
      h_t   = logistic z_u         -- current activation.
                                   -- *Need to include somewhere in output!*
      z_h   = (vB + vN #> h_t)     -- next layer of weights
      y_t   = logistic z_h         -- current final activation i.e. output
      -- Here, E = (y_t - tgt_t) ^ 2 is implied.
      dEdz = logistic' z_h * (y_t - tgt_t) -- this vector reports, for each 
                                            -- component of the final activation (y_t)_j,
                                            -- how much a unit "bump" to it will impact the
                                            -- matching component (E_t)_j of the error, 
                                            -- which is not explicitly calculated here.
      -- Bump the preceding-weights bias-vector vB in the opposite direction
      -- of dEdy, & (per component) in proportion with the impact that its change
      -- has on the error E. This is (- scale rate dEdy) (we scale by learning rate)
      vB'   = vB - scale rate dEdz
      -- Bump every matrix entry in vN by the appropriate amount.
      -- Row #1's matrix-entry bumps are given by transposing the layer input (h_t)'s
      -- entries by (- scale rate dEdy)_1, then transposing (so that connections from the
      -- 1st component of h_t will be reinforced appropriately.) 
      -- Do this for all entries of "d_vB / d_y = -scale rate dEdy", and you will get
      vN'   = vN + ((- scale rate dEdz) `outer` h_t)
      -- The above two terms vB', vN' specified how to "bump" the entries of the V-layer
      -- bias & weights matrix, given what the layer input h_t currently is.
      -- Now, figure out how to "bump" the entries of the V-layer input h_t,
      -- given what the current V-layer weights vB / vN are.
      -- In fact, only vN will play any role.
      dh_t  = tr vN #> dEdz -- should it be (- scale rate dEdy)?
      -- I think that the learning rate factor is omitted when back-propagating 
      -- a new error to the preceding layer, because otherwise the n-th layer back
      -- would get a factor of (rate ^ n) -> 0 as n increases, causing only the
      -- very few final layers to really be modified.

      -- Now, backpropagate this desired change in h_t to train the wts in U, W.
      -- Just as we previously had      "dy := - scale rate dEdy"
      -- so also will we now have       "dEdh := logistic' h_t * dh_t"
      --                                "dh := - scale rate dEdh"
      -- Train both U (whose input was h_t_1) and W (whose input was x_t).
      -- The activation, h_t, on these linear maps' composite result is now
      -- the vector being nudged, because it was the input of the final layer.
      dEdh  = logistic' h_t * dh_t 

      uB'   = uB - scale rate dEdh
      uN'   = uN + ((- scale rate dEdh) `outer` h_t_1)
      wB'   = wB - scale rate dEdh 
      wN'   = wN + ((- scale rate dEdh) `outer` x_t)  -- x_t *not* h_t / h_t_1
  in  ( h_t,                  -- the new activation
        MkRNN                 -- the new RNN
          (W uB' uN') 
          ((W wB' wN') :&~
            O (W vB' vN')))

-- Sample function for using trainRNN one time.
f = do
    rnn_0 <- evalRandIO (randRNN 1 2 5)
    let h_rnn' = 
          trainRNN rnn_0 0.25
                   (vector $ fromIntegral <$> [0,0,0,0,0])
                   (vector $ fromIntegral <$> [0])
                   (vector $ fromIntegral <$> [1,4])
    return h_rnn'

-- Need function to product a random RNN.
-- The relevant dimensions need to be:

{-
h_t-1 -> U -> h_t -> V -> y_t
            ^
x     -> W /

For x (input vector)  - dim(x): m
    h (activation)    - dim(h): h 
    y (output vector) - dim(y): n 
we have that the matrix sizes should be
  W:  h * m
  U:  h * h
  V:  n * h

rnn@(MkRNN 
  u@(W uB uN) 
  n@(
    w@(W wB wN) :&~
    O v@(W vB vN))) 
-}
randRNN :: MonadRandom m => Int   -- input vector dim
                         -> Int   -- output vector dim
                         -> Int   -- hidden vector dim
                         -> m RNN 
randRNN i o h = do 
  u <- randomWeights h h 
  w <- randomWeights i h 
  v <- randomWeights h o 
  return $ MkRNN u (w :&~ O v)

{-
 A typical form of training data to take will be 
 a list [(inp, outp)] with inp  :: Vector Double
                           outp :: Vector Double
-}

foldRNNTraining :: Double 
                -> (Vector Double, RNN)    -- initial hidden state & RNN
                -> [(Vector Double,        -- training data
                     Vector Double)]
                -> IO (Vector Double, RNN) -- final hidden state & trained RNN
foldRNNTraining rate (h_0, rnn_0) pairs = 
  foldl (\io p@(x, tgt) -> do 
              (h, rnn) <- io
              putStrLn $ "Current prediction given: " ++ show x 
                         ++ " is: " ++ show (runRNN rnn x h)
              let h_rnn' = trainRNN rnn rate h x tgt 
              return h_rnn')
          (return (h_0, rnn_0))
          pairs

displayRNNTraining :: Double            -- training rate
                   -- -> Vector Double     -- initial activation vector h_0
                   -> [(Vector Double,  -- lists of training inputs
                        Vector Double)] -- paired with target outputs
                   -> IO ()             
displayRNNTraining rate pairs = do 
  -- get a random RNN 
  randRNN <- evalRandIO (randRNN 1 1 5)
  -- we will always use an initial activation h_0 of zeroes
  let h_0 = vector $ fromIntegral <$> (take 5 $ repeat 0)
  foldRNNTraining rate (h_0, randRNN) pairs
  return ()

-- no n-gram window this time, just inputs & outputs one at a time each.
rnn_training_data :: [(Vector Double,  -- 1 * 1
                       Vector Double)] -- 1 * 1
rnn_training_data = 
  let fin_pattern = take fin_pttrn_len inf_pattern 
      list_pairs =
        [ (take 1 sub_pattern, [sub_pattern !! 1]) 
            | j <- [0..(length fin_pattern - 2)],
              let sub_pattern = drop j fin_pattern ]
  in map (\(inp, out) -> (vector $ fromIntegral <$> inp, 
                          vector $ fromIntegral <$> out))
         list_pairs

-- Retry the simple RNN training using softmax.
-- But at the same time, the 1-vectors in the training data 
-- [(inp, out)] pairs should become one-hot encoding 3-vectors.
-- Likewise, we will want 'randRNN 3 3 5' (5 the size of the hidden layer.)
softmax :: Vector Double -> Vector Double 
softmax v = 
  let vexp = exp v :: Vector Double
  in  vexp / (scalar $ sumElements vexp)

-- Note that the derivative of (vector) softmax is matrix-valued
softmax' :: Vector Double -> Matrix Double 
softmax' v = 
  let s = softmax v 
  in  diag s - s `outer` s

runRNN_softmax :: RNN 
               -> Vector Double -- input vector
               -> Vector Double -- previous-step activation
               -> Vector Double -- prediction
runRNN_softmax rnn@(MkRNN 
  u@(W uB uN) 
  n@(
    w@(W wB wN) :&~
    O v@(W vB vN))) x_t h_t_1 = 
    let z_u   = (uB + uN #> h_t_1) + 
                (wB + wN #> x_t)     
        h_t   = logistic z_u         
        z_h   = (vB + vN #> h_t)     
        y_t   = softmax z_h         
    in  y_t  -- ^ softmax usually, for tagging problems & similar 
             --   (multiple way classification)

trainRNN_softmax  :: RNN 
                  -> Double        
                  -> Vector Double 
                  -> Vector Double 
                  -> Vector Double 
                  -> (Vector Double, RNN) 
trainRNN_softmax rnn@(MkRNN 
  u@(W uB uN) 
  n@(
    w@(W wB wN) :&~
    O v@(W vB vN))) rate h_t_1 x_t tgt_t = 
  let z_u   = (uB + uN #> h_t_1) + 
              (wB + wN #> x_t)    
      h_t   = logistic z_u        
      z_h   = (vB + vN #> h_t)     
      y_t   = softmax z_h                   -- <-     
      dEdz  = softmax' z_h #> (y_t - tgt_t)  -- <- 
      vB'   = vB - scale rate dEdz
      vN'   = vN + ((- scale rate dEdz) `outer` h_t)
      dh_t  = tr vN #> dEdz 
      dEdh  = logistic' h_t * dh_t 

      uB'   = uB - scale rate dEdh
      uN'   = uN + ((- scale rate dEdh) `outer` h_t_1)
      wB'   = wB - scale rate dEdh 
      wN'   = wN + ((- scale rate dEdh) `outer` x_t)  
  in  ( h_t,                  
        MkRNN                 
          (W uB' uN') 
          ((W wB' wN') :&~
            O (W vB' vN')))

foldRNNTraining_softmax :: Double 
                        -> (Vector Double, RNN)    -- initial hidden state & RNN
                        -> [(Vector Double,        -- training data
                             Vector Double)]
                        -> IO (Vector Double, RNN) -- final hidden state & trained RNN
foldRNNTraining_softmax rate (h_0, rnn_0) pairs = 
  foldl (\io p@(x, tgt) -> do 
              (h, rnn) <- io
              putStrLn $ "Current prediction given: " ++ show x 
                         ++ " is: " ++ show (runRNN_softmax rnn x h)
              let h_rnn' = trainRNN_softmax rnn rate h x tgt 
              return h_rnn')
          (return (h_0, rnn_0))
          pairs

displayRNNTraining_softmax :: Double            
                           -> [(Vector Double,  -- lists of training inputs
                                Vector Double)] -- paired with target outputs
                           -> IO ()             
displayRNNTraining_softmax rate pairs = do 
  -- get a random RNN 
  randRNN <- evalRandIO (randRNN 3 3 7)
  -- we will always use an initial activation h_0 of zeroes
  let h_0 = vector $ fromIntegral <$> (take 7 $ repeat 0)
  foldRNNTraining_softmax rate (h_0, randRNN) pairs
  return ()

-- 
to_onehot :: [Int] -> Vector Double 
to_onehot m = case m of 
  [0] -> vector $ fromIntegral <$> [1, 0, 0]
  [1] -> vector $ fromIntegral <$> [0, 1, 0]
  [2] -> vector $ fromIntegral <$> [0, 0, 1]
  -- so that to_onehot won't be partial. Not using this branch
  _   -> vector $ fromIntegral <$> [0, 0, 0]

rnn_training_data_onehot :: [(Vector Double,  -- 3 * 1
                              Vector Double)] -- 3 * 1
rnn_training_data_onehot = 
  let fin_pattern = take fin_pttrn_len inf_pattern 
      list_pairs =
        [ (take 1 sub_pattern, [sub_pattern !! 1]) 
            | j <- [0..(length fin_pattern - 2)],
              let sub_pattern = drop j fin_pattern ]
  in map (\(inp, out) -> (to_onehot inp, to_onehot out))
         list_pairs

{-
 To see these examples in action:

 > displayTraining 0.25 training_data
 > displayRNNTraining 0.25 rnn_training_data
 > displayRNNTraining_softmax 0.25 rnn_training_data_onehot

-}



(!!?) :: [a] -> Int -> Maybe a
xs !!? i = listToMaybe (drop i xs)

main :: IO ()
main = return ()
