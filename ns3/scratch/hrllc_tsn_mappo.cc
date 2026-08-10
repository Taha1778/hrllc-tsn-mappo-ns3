#include "ns3/core-module.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ns3;

namespace
{

// Data exchanged between Python and C++, plus state carried during one
// simulation. These structures keep neural weights, devices, and frames explicit.
struct Matrix
{
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<double> data;

  double
  operator() (std::size_t r, std::size_t c) const
  {
    return data.at (r * cols + c);
  }
};

struct ActorWeights
{
  Matrix w1;
  std::vector<double> b1;
  Matrix w2;
  std::vector<double> b2;
  Matrix wq;
  std::vector<double> bq;
  Matrix wn;
  std::vector<double> bn;
};

struct ModelWeights
{
  uint32_t k = 0;
  uint32_t obsDim = 0;
  uint32_t stateDim = 0;
  uint32_t hiddenDim = 0;
  uint32_t qMax = 0;
  uint32_t nActionDim = 0;
  std::vector<ActorWeights> actors;
};

struct FlowType
{
  double periodMs;
  double latencyMs;
  double bytes;
  double epsilon;
};

struct Equipment
{
  double x = 0.0;
  double y = 0.0;
  double directionRad = 0.0;
  double speedMps = 0.0;
  double shadowDb = 0.0;
  double lambda = 0.0;
  double lambdaNorm = 0.0;
  double prevPowerW = 0.0;
  uint32_t prevRetransmissions = 0;
  uint32_t flowType = 0;
};

struct Frame
{
  uint32_t k = 0;
  uint32_t j = 0;
  double tC = 0.0;
  double rhoBytes = 0.0;
  double latencyLimitMs = 0.0;
  double epsilonLimit = 0.0;
  double lCs = 0.0;
  double tPrimeS = 0.0;
  double tStartSn = 0.0;
  double lSn = 0.0;
  double tN = 0.0;
  double tD = 0.0;
  double lNdBudget = 0.0;
  double lH = 0.0;
  double lW = 0.0;
  double lE2e = 0.0;
  double failureProbability = 1.0;
  bool latencyOk = false;
  bool reliabilityOk = false;
  bool success = false;
};

// Weight-file reader: Python exports actor parameters in a compact text format.
// C++ validates labels and shapes before using them for inference.
std::string
ExpectToken (std::istream& input)
{
  std::string token;
  if (!(input >> token))
    {
      throw std::runtime_error ("Unexpected end of weight file");
    }
  return token;
}

void
ExpectLabel (std::istream& input, const std::string& expected)
{
  const std::string got = ExpectToken (input);
  if (got != expected)
    {
      throw std::runtime_error ("Expected weight label '" + expected + "', got '" + got + "'");
    }
}

uint32_t
ReadUInt (std::istream& input, const std::string& label)
{
  ExpectLabel (input, label);
  uint32_t value;
  input >> value;
  if (!input)
    {
      throw std::runtime_error ("Could not read integer for " + label);
    }
  return value;
}

Matrix
ReadMatrix (std::istream& input, const std::string& label)
{
  ExpectLabel (input, label);
  Matrix matrix;
  input >> matrix.rows >> matrix.cols;
  if (!input)
    {
      throw std::runtime_error ("Could not read matrix shape for " + label);
    }
  matrix.data.resize (matrix.rows * matrix.cols);
  for (double& value : matrix.data)
    {
      input >> value;
      if (!input)
        {
          throw std::runtime_error ("Could not read matrix values for " + label);
        }
    }
  return matrix;
}

std::vector<double>
ReadVector (std::istream& input, const std::string& label)
{
  ExpectLabel (input, label);
  uint32_t size;
  input >> size;
  if (!input)
    {
      throw std::runtime_error ("Could not read vector size for " + label);
    }
  std::vector<double> vector (size);
  for (double& value : vector)
    {
      input >> value;
      if (!input)
        {
          throw std::runtime_error ("Could not read vector values for " + label);
        }
    }
  return vector;
}

ModelWeights
LoadWeights (const std::string& path)
{
  std::ifstream input (path);
  if (!input)
    {
      throw std::runtime_error ("Could not open weights file: " + path);
    }

  ExpectLabel (input, "HRLLC_TSN_MAPPO_WEIGHTS_V1");
  ModelWeights weights;
  weights.k = ReadUInt (input, "K");
  weights.obsDim = ReadUInt (input, "OBS_DIM");
  weights.stateDim = ReadUInt (input, "STATE_DIM");
  weights.hiddenDim = ReadUInt (input, "HIDDEN_DIM");
  weights.qMax = ReadUInt (input, "Q_MAX");
  weights.nActionDim = ReadUInt (input, "N_ACTION_DIM");
  weights.actors.resize (weights.k);

  for (uint32_t k = 0; k < weights.k; ++k)
    {
      ExpectLabel (input, "ACTOR");
      uint32_t actorIndex;
      input >> actorIndex;
      if (actorIndex != k)
        {
          throw std::runtime_error ("Actor index mismatch in weights");
        }
      auto& actor = weights.actors[k];
      actor.w1 = ReadMatrix (input, "W1");
      actor.b1 = ReadVector (input, "B1");
      actor.w2 = ReadMatrix (input, "W2");
      actor.b2 = ReadVector (input, "B2");
      actor.wq = ReadMatrix (input, "WQ");
      actor.bq = ReadVector (input, "BQ");
      actor.wn = ReadMatrix (input, "WN");
      actor.bn = ReadVector (input, "BN");
    }

  return weights;
}

// Minimal neural-network inference implementation. It mirrors Python's two
// ReLU layers and two softmax action heads without depending on a ML runtime.
std::vector<double>
Linear (const Matrix& w, const std::vector<double>& b, const std::vector<double>& x)
{
  if (w.cols != x.size () || w.rows != b.size ())
    {
      throw std::runtime_error ("Linear layer shape mismatch");
    }
  std::vector<double> y (w.rows, 0.0);
  for (std::size_t r = 0; r < w.rows; ++r)
    {
      double sum = b[r];
      for (std::size_t c = 0; c < w.cols; ++c)
        {
          sum += w (r, c) * x[c];
        }
      y[r] = sum;
    }
  return y;
}

std::vector<double>
Relu (std::vector<double> x)
{
  for (double& value : x)
    {
      value = std::max (0.0, value);
    }
  return x;
}

std::vector<double>
Softmax (const std::vector<double>& logits)
{
  const double maxLogit = *std::max_element (logits.begin (), logits.end ());
  std::vector<double> probs (logits.size ());
  double denom = 0.0;
  for (std::size_t i = 0; i < logits.size (); ++i)
    {
      probs[i] = std::exp (logits[i] - maxLogit);
      denom += probs[i];
    }
  for (double& value : probs)
    {
      value /= denom;
    }
  return probs;
}

std::vector<double>
MaskedSoftmax (const std::vector<double>& logits, uint32_t validCount, double maskLogit)
{
  std::vector<double> masked = logits;
  for (uint32_t i = validCount; i < masked.size (); ++i)
    {
      masked[i] = maskLogit;
    }
  return Softmax (masked);
}

std::pair<std::vector<double>, std::vector<double>>
ForwardActor (const ActorWeights& actor,
              const std::vector<double>& obs,
              uint32_t validNCount,
              double maskLogit)
{
  const auto h1 = Relu (Linear (actor.w1, actor.b1, obs));
  const auto h2 = Relu (Linear (actor.w2, actor.b2, h1));
  return {Softmax (Linear (actor.wq, actor.bq, h2)),
          MaskedSoftmax (Linear (actor.wn, actor.bn, h2), validNCount, maskLogit)};
}

uint32_t
SampleCategorical (const std::vector<double>& probs, Ptr<UniformRandomVariable> rng)
{
  const double u = rng->GetValue (0.0, 1.0);
  double cumulative = 0.0;
  for (uint32_t i = 0; i < probs.size (); ++i)
    {
      cumulative += probs[i];
      if (u <= cumulative)
        {
          return i;
        }
    }
  return static_cast<uint32_t> (probs.size () - 1);
}

uint32_t
ArgMaxAction (const std::vector<double>& probs)
{
  return static_cast<uint32_t> (std::distance (probs.begin (),
                                               std::max_element (probs.begin (), probs.end ())));
}

// Physical-layer helper functions: unit conversions and the integer-shape
// Gamma CDF used by the paper's retransmission failure-probability expression.
double
Clamp (double value, double lo, double hi)
{
  return std::max (lo, std::min (hi, value));
}

double
DbToLinear (double db)
{
  return std::pow (10.0, db / 10.0);
}

double
DbmHzToWHz (double dbmHz)
{
  return std::pow (10.0, (dbmHz - 30.0) / 10.0);
}

double
GammaCdfIntegerShape (uint32_t shape, double x)
{
  if (x <= 0.0)
    {
      return 0.0;
    }
  double term = 1.0;
  double sum = 1.0;
  for (uint32_t m = 1; m < shape; ++m)
    {
      term *= x / static_cast<double> (m);
      sum += term;
    }
  const double cdf = 1.0 - std::exp (-x) * sum;
  return Clamp (cdf, 0.0, 1.0);
}

std::vector<double>
ParseCommaSeparatedDoubles (const std::string& text)
{
  std::vector<double> values;
  std::stringstream stream (text);
  std::string token;
  while (std::getline (stream, token, ','))
    {
      if (!token.empty ())
        {
          values.push_back (std::stod (token));
        }
    }
  return values;
}

class HrlccTsnMappoSimulation
{
public:
  // The constructor receives all experiment settings from Python/JSON. It
  // initializes random variables, device state, flow definitions, and CSV output.
  HrlccTsnMappoSimulation (ModelWeights weights,
                           std::string csvPath,
                           uint32_t iterations,
                           uint32_t seed,
                           uint32_t k,
                           double radiusM,
                           double speedMinMps,
                           double speedMaxMps,
                           double hyperperiodMs,
                           double tsnRateMbps,
                           double hrllcBandwidthMbps,
                           double carrierFrequencyGhz,
                           double miniSlotMs,
                           double retransmissionMs,
                           double retransmissionGuardMs,
                           uint32_t hopCount,
                           double flow1PeriodMs,
                           double flow1LatencyMs,
                           double flow1Bytes,
                           double flow1Epsilon,
                           double flow2PeriodMs,
                           double flow2LatencyMs,
                           double flow2Bytes,
                           double flow2Epsilon,
                           double pH,
                           double gammaThresholdDb,
                           double shadowFadingStdDb,
                           double smallScaleFadingRate,
                           double thermalNoiseDensityDbmHz,
                           double pathLossC,
                           double pathLossD0,
                           double pathLossD1,
                           double boundaryTurnMinRad,
                           double boundaryTurnMaxRad,
                           double minimumDistanceM,
                           double lambdaNormalizationOffsetDb,
                           double lambdaNormalizationScaleDb,
                           double lambdaNormalizationMin,
                           double lambdaNormalizationMax,
                           double softmaxMaskLogit,
                           double probabilityFloor,
                           double denominatorFloor,
                           double reliabilityPenaltyWeight,
                           bool shadowFadingPerStep,
                           bool conservativeRetryFeasibility,
                           bool randomGcl,
                           bool simpleReward,
                           std::vector<double> referenceGains,
                           bool deterministic,
                           uint32_t runIndex)
    : m_weights (std::move (weights)),
      m_csvPath (std::move (csvPath)),
      m_iterations (iterations),
      m_seed (seed),
      m_k (k),
      m_radiusM (radiusM),
      m_speedMinMps (speedMinMps),
      m_speedMaxMps (speedMaxMps),
      m_hyperperiodMs (hyperperiodMs),
      m_tsnRateMbps (tsnRateMbps),
      m_hrllcBandwidthMbps (hrllcBandwidthMbps),
      m_carrierFrequencyGhz (carrierFrequencyGhz),
      m_miniSlotMs (miniSlotMs),
      m_retransmissionMs (retransmissionMs),
      m_retransmissionGuardMs (retransmissionGuardMs),
      m_hopCount (hopCount),
      m_totalPowerW (pH),
      m_gammaThresholdDb (gammaThresholdDb),
      m_shadowFadingStdDb (shadowFadingStdDb),
      m_smallScaleFadingRate (smallScaleFadingRate),
      m_thermalNoiseDensityDbmHz (thermalNoiseDensityDbmHz),
      m_pathLossC (pathLossC),
      m_pathLossD0 (pathLossD0),
      m_pathLossD1 (pathLossD1),
      m_boundaryTurnMinRad (boundaryTurnMinRad),
      m_boundaryTurnMaxRad (boundaryTurnMaxRad),
      m_minimumDistanceM (minimumDistanceM),
      m_lambdaNormalizationOffsetDb (lambdaNormalizationOffsetDb),
      m_lambdaNormalizationScaleDb (lambdaNormalizationScaleDb),
      m_lambdaNormalizationMin (lambdaNormalizationMin),
      m_lambdaNormalizationMax (lambdaNormalizationMax),
      m_softmaxMaskLogit (softmaxMaskLogit),
      m_probabilityFloor (probabilityFloor),
      m_denominatorFloor (denominatorFloor),
      m_reliabilityPenaltyWeight (reliabilityPenaltyWeight),
       m_shadowFadingPerStep (shadowFadingPerStep),
       m_conservativeRetryFeasibility (conservativeRetryFeasibility),
       m_randomGcl (randomGcl),
       m_simpleReward (simpleReward),
      m_referenceGains (std::move (referenceGains)),
      m_deterministic (deterministic),
      m_runIndex (runIndex),
      m_stdRng (seed)
  {
    if (m_k != m_weights.k)
      {
        throw std::runtime_error ("K mismatch between command line and weight file");
      }
    m_uniform = CreateObject<UniformRandomVariable> ();
    m_uniform->SetStream (seed + 17);
    m_normal = std::normal_distribution<double> (0.0, m_shadowFadingStdDb);
    m_exponential = std::exponential_distribution<double> (m_smallScaleFadingRate);
    m_flows = {
        {flow1PeriodMs, flow1LatencyMs, flow1Bytes, flow1Epsilon},
        {flow2PeriodMs, flow2LatencyMs, flow2Bytes, flow2Epsilon},
    };
    InitializeEquipment ();
    OpenCsv ();
  }

  ~HrlccTsnMappoSimulation ()
  {
    if (m_csv.is_open ())
      {
        m_csv.close ();
      }
  }

  void
  Run ()
  {
    // ns-3 schedules one independent decision/scheduling cycle per hyperperiod.
    for (uint32_t step = 0; step < m_iterations; ++step)
      {
        Simulator::Schedule (MilliSeconds (step * m_hyperperiodMs),
                             &HrlccTsnMappoSimulation::RunStep,
                             this,
                             step);
      }
    Simulator::Run ();
    Simulator::Destroy ();
  }

private:
  // Initial device placement follows the configured circular deployment. Each
  // device retains its flow type and previous action for the next observation.
  void
  InitializeEquipment ()
  {
    m_equipment.resize (m_k);
    for (uint32_t k = 0; k < m_k; ++k)
      {
        const double r = m_radiusM * std::sqrt (m_uniform->GetValue (0.0, 1.0));
        const double theta = m_uniform->GetValue (0.0, 2.0 * M_PI);
        auto& eq = m_equipment[k];
        eq.x = r * std::cos (theta);
        eq.y = r * std::sin (theta);
        eq.directionRad = m_uniform->GetValue (0.0, 2.0 * M_PI);
        eq.speedMps = m_uniform->GetValue (m_speedMinMps, m_speedMaxMps);
        eq.shadowDb = m_normal (m_stdRng);
        eq.prevPowerW = m_totalPowerW / static_cast<double> (m_k);
        eq.prevRetransmissions = 0;
        eq.flowType = k < m_k / 2 ? 0 : 1;
      }
  }

  void
  OpenCsv ()
  {
    // The CSV is the interface back to Python: it contains actions, outcomes,
    // rewards, and the state features needed to reconstruct a PPO trajectory.
    m_csv.open (m_csvPath);
    if (!m_csv)
      {
        throw std::runtime_error ("Could not open output CSV: " + m_csvPath);
      }

    m_csv << "run,step,agent,action_q_index,action_q_factor,action_n_index,retransmissions,action_q_logp,action_n_logp,"
          << "power_w,reward,frames,successes,latency_violations,reliability_violations,"
          << "avg_e2e_ms,avg_lw_ms,avg_failure_probability,lambda_raw";
    for (uint32_t i = 0; i < m_k; ++i)
      {
        m_csv << ",lambda_norm_" << i;
      }
    for (uint32_t i = 0; i < m_k; ++i)
      {
        m_csv << ",prev_power_norm_" << i;
      }
    for (uint32_t i = 0; i < m_k; ++i)
      {
        m_csv << ",prev_n_norm_" << i;
      }
    m_csv << "\n";
  }

  void
  RunStep (uint32_t step)
  {
    // One MAPPO environment step: update channels, infer actions, allocate the
    // fixed power budget, schedule frames, calculate reward, and record results.
    UpdateMobilityAndChannel ();

    std::vector<uint32_t> qIndex (m_k);
    std::vector<uint32_t> qFactor (m_k);
    std::vector<uint32_t> nIndex (m_k);
    std::vector<uint32_t> retransmissions (m_k);
    std::vector<double> powerW (m_k);
    std::vector<double> qLogp (m_k);
    std::vector<double> nLogp (m_k);

    std::vector<double> lambdaNorms;
    std::vector<double> prevPowerNorms;
    std::vector<double> prevNNorms;
    BuildStateVectors (lambdaNorms, prevPowerNorms, prevNNorms);

    for (uint32_t k = 0; k < m_k; ++k)
      {
        std::vector<double> obs = lambdaNorms;
        obs.push_back (prevPowerNorms[k]);
        obs.push_back (prevNNorms[k]);
        const auto probs = ForwardActor (
            m_weights.actors[k], obs, ValidRetransmissionActionCount (k), m_softmaxMaskLogit);
        qIndex[k] = m_deterministic ? ArgMaxAction (probs.first) : SampleCategorical (probs.first, m_uniform);
        nIndex[k] = m_deterministic ? ArgMaxAction (probs.second) : SampleCategorical (probs.second, m_uniform);
        qFactor[k] = qIndex[k] + 1;
        retransmissions[k] = nIndex[k];
        qLogp[k] = std::log (std::max (probs.first[qIndex[k]], m_probabilityFloor));
        nLogp[k] = std::log (std::max (probs.second[nIndex[k]], m_probabilityFloor));
      }

    const double qSum =
        std::accumulate (qFactor.begin (), qFactor.end (), 0.0);
    for (uint32_t k = 0; k < m_k; ++k)
      {
        powerW[k] = (static_cast<double> (qFactor[k]) / qSum) * m_totalPowerW;
      }

    std::vector<Frame> frames = GenerateAndScheduleFrames (powerW, retransmissions);

    std::vector<uint32_t> framesByAgent (m_k, 0);
    std::vector<uint32_t> successesByAgent (m_k, 0);
    std::vector<uint32_t> latencyViolationsByAgent (m_k, 0);
    std::vector<uint32_t> reliabilityViolationsByAgent (m_k, 0);
    std::vector<double> e2eSumByAgent (m_k, 0.0);
    std::vector<double> lwSumByAgent (m_k, 0.0);
    std::vector<double> failureSumByAgent (m_k, 0.0);

    double reward = 0.0;
    for (const auto& frame : frames)
      {
        framesByAgent[frame.k]++;
        successesByAgent[frame.k] += frame.success ? 1 : 0;
        latencyViolationsByAgent[frame.k] += frame.latencyOk ? 0 : 1;
        reliabilityViolationsByAgent[frame.k] += frame.reliabilityOk ? 0 : 1;
        e2eSumByAgent[frame.k] += frame.lE2e;
        lwSumByAgent[frame.k] += frame.lW;
        failureSumByAgent[frame.k] += frame.failureProbability;

        // Equations (27)-(29): reward is zero only when every frame meets
        // both requirements; late or unreliable frames add a negative penalty.
        if (m_simpleReward)
          {
            reward += frame.success ? 0.0 : -1.0;
          }
        else if (frame.lW < 0.0)
          {
            reward += frame.lW;
          }
        if (frame.failureProbability > frame.epsilonLimit)
          {
            reward += m_reliabilityPenaltyWeight *
                      (frame.epsilonLimit /
                       std::max (frame.failureProbability, m_denominatorFloor) - 1.0);
          }
      }

    for (uint32_t k = 0; k < m_k; ++k)
      {
        const double denom = std::max (1u, framesByAgent[k]);
        m_csv << m_runIndex << "," << step << "," << k << "," << qIndex[k] << ","
              << qFactor[k] << "," << nIndex[k] << "," << retransmissions[k] << ","
              << qLogp[k] << "," << nLogp[k] << ","
              << std::setprecision (12) << powerW[k] << "," << reward << ","
              << framesByAgent[k] << "," << successesByAgent[k] << ","
              << latencyViolationsByAgent[k] << "," << reliabilityViolationsByAgent[k] << ","
              << e2eSumByAgent[k] / denom << "," << lwSumByAgent[k] / denom << ","
              << failureSumByAgent[k] / denom << "," << m_equipment[k].lambda;
        for (double value : lambdaNorms)
          {
            m_csv << "," << value;
          }
        for (double value : prevPowerNorms)
          {
            m_csv << "," << value;
          }
        for (double value : prevNNorms)
          {
            m_csv << "," << value;
          }
        m_csv << "\n";
      }

    for (uint32_t k = 0; k < m_k; ++k)
      {
        m_equipment[k].prevPowerW = powerW[k];
        m_equipment[k].prevRetransmissions = retransmissions[k];
      }
  }

  void
  UpdateMobilityAndChannel ()
  {
    // Move devices, keep them inside the circular plant, and derive large-scale
    // channel gain from path loss plus the configured shadow-fading model.
    const double dtSeconds = m_hyperperiodMs / 1000.0;
    for (auto& eq : m_equipment)
      {
        eq.x += eq.speedMps * dtSeconds * std::cos (eq.directionRad);
        eq.y += eq.speedMps * dtSeconds * std::sin (eq.directionRad);

        double distanceFromCenter = std::sqrt (eq.x * eq.x + eq.y * eq.y);
        if (distanceFromCenter >= m_radiusM)
          {
            const double scale = (m_radiusM / std::max (distanceFromCenter, 1e-12)) * 0.99;
            eq.x *= scale;
            eq.y *= scale;
            eq.directionRad = m_uniform->GetValue (0.0, 2.0 * M_PI);
            eq.speedMps = m_uniform->GetValue (m_speedMinMps, m_speedMaxMps);
          }
        if (m_shadowFadingPerStep)
          {
            eq.shadowDb = m_normal (m_stdRng);
          }

        const double d = std::max (m_minimumDistanceM, std::sqrt (eq.x * eq.x + eq.y * eq.y));
        double pathLossDb;
        if (d > m_pathLossD1)
          {
            pathLossDb = -PathLossC () - 35.0 * std::log10 (d);
          }
        else if (d > m_pathLossD0)
          {
            pathLossDb = -PathLossC () - 15.0 * std::log10 (m_pathLossD1) - 20.0 * std::log10 (d);
          }
        else
          {
            pathLossDb = -PathLossC () - 15.0 * std::log10 (m_pathLossD1) -
                         20.0 * std::log10 (m_pathLossD0);
          }
        const double lambdaDb = pathLossDb + eq.shadowDb;
        eq.lambda = DbToLinear (lambdaDb);
        eq.lambdaNorm = Clamp ((lambdaDb + m_lambdaNormalizationOffsetDb) /
                                   m_lambdaNormalizationScaleDb,
                               m_lambdaNormalizationMin,
                               m_lambdaNormalizationMax);
      }
  }

  void
  BuildStateVectors (std::vector<double>& lambdaNorms,
                     std::vector<double>& prevPowerNorms,
                     std::vector<double>& prevNNorms) const
  {
    lambdaNorms.clear ();
    prevPowerNorms.clear ();
    prevNNorms.clear ();
    for (const auto& eq : m_equipment)
      {
        lambdaNorms.push_back (eq.lambdaNorm);
      }
    for (const auto& eq : m_equipment)
      {
        prevPowerNorms.push_back (eq.prevPowerW / std::max (m_totalPowerW, 1e-12));
      }
    for (const auto& eq : m_equipment)
      {
        prevNNorms.push_back (static_cast<double> (eq.prevRetransmissions) /
                              static_cast<double> (std::max (1u, m_weights.nActionDim - 1)));
      }
  }

  std::vector<Frame>
  GenerateAndScheduleFrames (const std::vector<double>& powerW,
                             const std::vector<uint32_t>& retransmissions)
  {
    // Build periodic traffic, serialize it through TSN, then calculate wireless
    // service time and determine each frame's latency and reliability outcome.
    std::vector<Frame> frames;
    for (uint32_t k = 0; k < m_k; ++k)
      {
        const auto& flow = m_flows[m_equipment[k].flowType];
        const uint32_t framesPerHyperperiod =
            static_cast<uint32_t> (std::round (m_hyperperiodMs / flow.periodMs));
        for (uint32_t j = 0; j < framesPerHyperperiod; ++j)
          {
            Frame frame;
            frame.k = k;
            frame.j = j;
            frame.tC = j * flow.periodMs;
            frame.rhoBytes = flow.bytes;
            frame.latencyLimitMs = flow.latencyMs;
            frame.epsilonLimit = flow.epsilon;
            frame.lCs = m_hopCount * flow.bytes / TsnDataUnitsPerMs ();
            frame.tPrimeS = frame.tC + frame.lCs;
            frame.tD = frame.tC + flow.latencyMs - flow.bytes / TsnDataUnitsPerMs ();
            frames.push_back (frame);
          }
      }

    if (m_randomGcl)
      {
        std::shuffle (frames.begin (), frames.end (), m_stdRng);
      }
    else
      {
        std::sort (frames.begin (), frames.end (), [this] (const Frame& a, const Frame& b) {
      if (std::abs (a.tPrimeS - b.tPrimeS) > 1e-12)
        {
          return a.tPrimeS < b.tPrimeS;
        }
      return m_equipment[a.k].lambda < m_equipment[b.k].lambda;
        });
      }

    double tsnLinkAvailable = 0.0;
    for (auto& frame : frames)
      {
        frame.tStartSn = std::max (tsnLinkAvailable, frame.tPrimeS);
        const double switchToNwTx = frame.rhoBytes / TsnDataUnitsPerMs ();
        frame.lSn = frame.tStartSn + switchToNwTx - frame.tPrimeS;
        frame.tN = frame.tPrimeS + frame.lSn;
        tsnLinkAvailable = frame.tStartSn + switchToNwTx;

        const uint32_t k = frame.k;
        const double snrRate = InstantaneousSnr (k, powerW[k], 1);
        const double rAirDataUnitsPerMs =
            (BandwidthHzPerUser () / 1000.0) * std::log2 (1.0 + snrRate);
        const double txSlots =
            std::ceil (frame.rhoBytes / std::max (rAirDataUnitsPerMs, 1e-12) / m_miniSlotMs);
        uint32_t attempts = retransmissions[k] + 1;
        bool decoded = false;
        for (uint32_t attempt = 1; attempt <= retransmissions[k] + 1; ++attempt)
          {
            if (InstantaneousSnr (k, powerW[k], attempt) >= DbToLinear (m_gammaThresholdDb))
              {
                attempts = attempt;
                decoded = true;
                break;
              }
          }
        const uint32_t actualRetransmissions = decoded ? attempts - 1 : retransmissions[k];
        frame.lH = attempts * txSlots * m_miniSlotMs +
                   actualRetransmissions * m_retransmissionMs;
        frame.failureProbability = FailureProbability (k, powerW[k], retransmissions[k]);
        frame.lNdBudget = frame.tD - frame.tN;
        frame.lW = frame.lNdBudget - frame.lH;
        frame.lE2e = frame.tD + frame.rhoBytes / TsnDataUnitsPerMs () - frame.tC;
        frame.latencyOk = frame.lE2e <= frame.latencyLimitMs && frame.lW >= 0.0;
        frame.reliabilityOk = frame.failureProbability <= frame.epsilonLimit;
        frame.success = frame.latencyOk && frame.reliabilityOk;
      }
    return frames;
  }

  double
  TsnDataUnitsPerMs () const
  {
    return m_tsnRateMbps * 1000.0;
  }

  double
  BandwidthHzPerUser () const
  {
    return (m_hrllcBandwidthMbps * 1e6) / static_cast<double> (m_k);
  }

  double
  PathLossC () const
  {
    return m_pathLossC + 20.0 * std::log10 (m_carrierFrequencyGhz / 3.5);
  }

  double
  InstantaneousSnr (uint32_t k, double powerW, uint32_t attempt)
  {
    const double fadingPower = std::max (ReferenceGain (k, attempt), m_probabilityFloor);
    const double noise = BandwidthHzPerUser () * DbmHzToWHz (m_thermalNoiseDensityDbmHz);
    return powerW * m_equipment[k].lambda * fadingPower / std::max (noise, m_denominatorFloor);
  }

  double
  ReferenceGain (uint32_t k, uint32_t attempt)
  {
    const uint32_t width = std::max (1u, m_weights.nActionDim);
    const std::size_t index = static_cast<std::size_t> (k) * width + (attempt - 1);
    if (attempt >= 1 && index < m_referenceGains.size ())
      {
        return m_referenceGains[index];
      }
    return m_exponential (m_stdRng);
  }

  double
  FailureProbability (uint32_t k, double powerW, uint32_t retransmissions) const
  {
    // Multiply the paper's Gamma-CDF terms across the first transmission and
    // selected retransmissions to obtain the end-to-end decoding failure risk.
    const double gammaPrime = DbToLinear (m_gammaThresholdDb);
    const double noise = BandwidthHzPerUser () * DbmHzToWHz (m_thermalNoiseDensityDbmHz);
    const double threshold =
        noise * gammaPrime / std::max (powerW * m_equipment[k].lambda, m_denominatorFloor);
    double failure = 1.0;
    for (uint32_t n = 1; n <= retransmissions + 1; ++n)
      {
        failure *= GammaCdfIntegerShape (n, threshold);
      }
    return Clamp (failure, 0.0, 1.0);
  }

  uint32_t
  ValidRetransmissionActionCount (uint32_t k) const
  {
    // Restrict retry actions to the paper's action space and, optionally, remove
    // retry counts that cannot fit within a conservative remaining latency budget.
    const auto& flow = m_flows[m_equipment[k].flowType];
    const double retransmissionBudgetMs = std::max (0.0, flow.latencyMs - m_retransmissionGuardMs);
    // Equation (25) explicitly defines N_k in {0, ..., floor(L_k / d_r)}.
    // This resolves the strict/inclusive boundary ambiguity in (20d).
    uint32_t maxN = static_cast<uint32_t> (
        std::floor (retransmissionBudgetMs / m_retransmissionMs + m_probabilityFloor));
    if (m_conservativeRetryFeasibility)
      {
        // Equation (7) requires L_ND - L_H >= 0. Reserve one mini-slot and
        // the maximum shared TSN egress service time before allowing retries.
        double worstTsnArrivalOffsetMs = 0.0;
        for (const auto& equipment : m_equipment)
          {
            worstTsnArrivalOffsetMs +=
                m_flows[equipment.flowType].bytes / TsnDataUnitsPerMs ();
          }
        const double feasibleRetryBudgetMs = std::max (
            0.0,
            flow.latencyMs - flow.bytes / TsnDataUnitsPerMs () -
                worstTsnArrivalOffsetMs - m_miniSlotMs);
        maxN = std::min (maxN, static_cast<uint32_t> (
            std::floor (feasibleRetryBudgetMs / m_retransmissionMs + m_probabilityFloor)));
      }
    return std::max (1u, std::min (maxN + 1, m_weights.nActionDim));
  }

private:
  // Runtime configuration and state. These values are populated by C++ command
  // line arguments, which Python derives from configs/default_config.json.
  ModelWeights m_weights;
  std::string m_csvPath;
  uint32_t m_iterations;
  uint32_t m_seed;
  uint32_t m_k;
  double m_radiusM;
  double m_totalPowerW;
  double m_gammaThresholdDb;
  double m_shadowFadingStdDb;
  double m_smallScaleFadingRate;
  double m_thermalNoiseDensityDbmHz;
  double m_pathLossC;
  double m_pathLossD0;
  double m_pathLossD1;
  double m_boundaryTurnMinRad;
  double m_boundaryTurnMaxRad;
  double m_minimumDistanceM;
  double m_lambdaNormalizationOffsetDb;
  double m_lambdaNormalizationScaleDb;
  double m_lambdaNormalizationMin;
  double m_lambdaNormalizationMax;
  double m_softmaxMaskLogit;
  double m_probabilityFloor;
  double m_denominatorFloor;
  double m_reliabilityPenaltyWeight;
  bool m_shadowFadingPerStep;
  bool m_conservativeRetryFeasibility;
  bool m_randomGcl;
  bool m_simpleReward;
  std::vector<double> m_referenceGains;
  bool m_deterministic;
  uint32_t m_runIndex;
  std::ofstream m_csv;

  double m_speedMinMps = 0.1;
  double m_speedMaxMps = 3.0;
  double m_hyperperiodMs = 10.0;
  double m_tsnRateMbps = 100.0;
  double m_hrllcBandwidthMbps = 20.0;
  double m_carrierFrequencyGhz = 3.5;
  double m_miniSlotMs = 0.25;
  double m_retransmissionMs = 1.0;
  double m_retransmissionGuardMs = 1.0;
  uint32_t m_hopCount = 4;
  std::vector<FlowType> m_flows;

  std::vector<Equipment> m_equipment;
  Ptr<UniformRandomVariable> m_uniform;
  std::mt19937_64 m_stdRng;
  std::normal_distribution<double> m_normal;
  std::exponential_distribution<double> m_exponential;
};

} // namespace

// ns-3 command-line entry point. Python supplies weights, output path, and all
// configuration values; this function constructs and runs one trajectory.
int
main (int argc, char* argv[])
{
  std::string weightsPath;
  std::string csvPath = "hrllc_tsn_mappo_results.csv";
  uint32_t iterations = 256;
  uint32_t seed = 1;
  uint32_t k = 8;
  double radiusM = 250.0;
  double speedMinMps = 0.1;
  double speedMaxMps = 3.0;
  double hyperperiodMs = 10.0;
  double tsnRateMbps = 100.0;
  double hrllcBandwidthMbps = 20.0;
  double carrierFrequencyGhz = 3.5;
  double miniSlotMs = 0.25;
  double retransmissionMs = 1.0;
  double retransmissionGuardMs = 0.0;
  uint32_t hopCount = 4;
  double flow1PeriodMs = 5.0;
  double flow1LatencyMs = 3.0;
  double flow1Bytes = 1000.0;
  double flow1Epsilon = 1e-7;
  double flow2PeriodMs = 10.0;
  double flow2LatencyMs = 5.0;
  double flow2Bytes = 1200.0;
  double flow2Epsilon = 1e-6;
  double totalPowerW = 40.0;
  double gammaThresholdDb = 5.0;
  double shadowFadingStdDb = 8.0;
  double smallScaleFadingRate = 1.0;
  double thermalNoiseDensityDbmHz = -174.0;
  double pathLossC = 35.7;
  double pathLossD0 = 10.0;
  double pathLossD1 = 50.0;
  double boundaryTurnMinRad = -0.7;
  double boundaryTurnMaxRad = 0.7;
  double minimumDistanceM = 1.0;
  double lambdaNormalizationOffsetDb = 140.0;
  double lambdaNormalizationScaleDb = 80.0;
  double lambdaNormalizationMin = -2.0;
  double lambdaNormalizationMax = 2.0;
  double softmaxMaskLogit = -1e100;
  double probabilityFloor = 1e-12;
  double denominatorFloor = 1e-300;
  double reliabilityPenaltyWeight = 1.0;
  bool shadowFadingPerStep = true;
  bool conservativeRetryFeasibility = true;
  bool randomGcl = false;
  bool simpleReward = false;
  std::string referenceGainsCsv;
  bool deterministic = false;
  uint32_t runIndex = 0;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("weights", "Path to MAPPO weight file", weightsPath);
  cmd.AddValue ("csv", "Path to output CSV file", csvPath);
  cmd.AddValue ("iterations", "Number of hyperperiods to simulate", iterations);
  cmd.AddValue ("seed", "Random seed", seed);
  cmd.AddValue ("K", "Number of industrial equipments", k);
  cmd.AddValue ("radius", "Circular factory radius in meters", radiusM);
  cmd.AddValue ("speedMin", "Minimum equipment speed in meters per second", speedMinMps);
  cmd.AddValue ("speedMax", "Maximum equipment speed in meters per second", speedMaxMps);
  cmd.AddValue ("hyperperiodMs", "Hyperperiod duration in milliseconds", hyperperiodMs);
  cmd.AddValue ("tsnRateMbps", "TSN link rate in Mbps", tsnRateMbps);
  cmd.AddValue ("hrllcBandwidthMbps", "Total HRLLC bandwidth in Mbps", hrllcBandwidthMbps);
  cmd.AddValue ("carrierFrequencyGhz", "Carrier frequency in GHz", carrierFrequencyGhz);
  cmd.AddValue ("miniSlotMs", "HRLLC mini-slot length in milliseconds", miniSlotMs);
  cmd.AddValue ("retransmissionMs", "HARQ retransmission latency in milliseconds", retransmissionMs);
  cmd.AddValue ("retransmissionGuardMs", "Latency guard reserved before retransmission budget", retransmissionGuardMs);
  cmd.AddValue ("hopCount", "Number of TSN switch hops", hopCount);
  cmd.AddValue ("flow1PeriodMs", "Type-1 flow period in milliseconds", flow1PeriodMs);
  cmd.AddValue ("flow1LatencyMs", "Type-1 flow maximum latency in milliseconds", flow1LatencyMs);
  cmd.AddValue ("flow1Bytes", "Type-1 flow data size in bytes", flow1Bytes);
  cmd.AddValue ("flow1Epsilon", "Type-1 flow allowed failure probability", flow1Epsilon);
  cmd.AddValue ("flow2PeriodMs", "Type-2 flow period in milliseconds", flow2PeriodMs);
  cmd.AddValue ("flow2LatencyMs", "Type-2 flow maximum latency in milliseconds", flow2LatencyMs);
  cmd.AddValue ("flow2Bytes", "Type-2 flow data size in bytes", flow2Bytes);
  cmd.AddValue ("flow2Epsilon", "Type-2 flow allowed failure probability", flow2Epsilon);
  cmd.AddValue ("pH", "Total HRLLC transmission power in watts", totalPowerW);
  cmd.AddValue ("gammaDb", "Minimum SNR required for correct decoding in dB", gammaThresholdDb);
  cmd.AddValue ("shadowFadingStdDb", "Shadow-fading standard deviation in dB", shadowFadingStdDb);
  cmd.AddValue ("smallScaleFadingRate", "Exponential fading-power rate", smallScaleFadingRate);
  cmd.AddValue ("thermalNoiseDensityDbmHz", "Thermal noise density in dBm/Hz", thermalNoiseDensityDbmHz);
  cmd.AddValue ("pathLossC", "Three-slope path-loss constant in dB", pathLossC);
  cmd.AddValue ("pathLossD0", "First path-loss breakpoint in meters", pathLossD0);
  cmd.AddValue ("pathLossD1", "Second path-loss breakpoint in meters", pathLossD1);
  cmd.AddValue ("boundaryTurnMinRad", "Minimum boundary-turn perturbation in radians", boundaryTurnMinRad);
  cmd.AddValue ("boundaryTurnMaxRad", "Maximum boundary-turn perturbation in radians", boundaryTurnMaxRad);
  cmd.AddValue ("minimumDistanceM", "Minimum BS-to-device distance in meters", minimumDistanceM);
  cmd.AddValue ("lambdaNormalizationOffsetDb", "Lambda normalization additive offset in dB", lambdaNormalizationOffsetDb);
  cmd.AddValue ("lambdaNormalizationScaleDb", "Lambda normalization divisor in dB", lambdaNormalizationScaleDb);
  cmd.AddValue ("lambdaNormalizationMin", "Lambda normalization lower clamp", lambdaNormalizationMin);
  cmd.AddValue ("lambdaNormalizationMax", "Lambda normalization upper clamp", lambdaNormalizationMax);
  cmd.AddValue ("softmaxMaskLogit", "Logit assigned to invalid retransmission actions", softmaxMaskLogit);
  cmd.AddValue ("probabilityFloor", "Positive floor for probability-like calculations", probabilityFloor);
  cmd.AddValue ("denominatorFloor", "Positive floor for denominator calculations", denominatorFloor);
  cmd.AddValue ("reliabilityPenaltyWeight", "Multiplier for reliability-violation reward penalties", reliabilityPenaltyWeight);
  cmd.AddValue ("shadowFadingPerStep", "Redraw log-normal shadow fading every time slot", shadowFadingPerStep);
  cmd.AddValue ("conservativeRetryFeasibility", "Mask retries that cannot meet a conservative L_W budget", conservativeRetryFeasibility);
  cmd.AddValue ("randomGcl", "Use random TSN gate-control ordering (MAPPO-R extension)", randomGcl);
  cmd.AddValue ("simpleReward", "Use unit penalty for each failed frame (MAPPO-S extension)", simpleReward);
  cmd.AddValue ("referenceGains", "Comma-separated Python gamma gains indexed by equipment then attempt", referenceGainsCsv);
  cmd.AddValue ("deterministic", "Use argmax actions instead of sampling", deterministic);
  cmd.AddValue ("runIndex", "Training-loop run index", runIndex);
  cmd.Parse (argc, argv);

  if (weightsPath.empty ())
    {
      std::cerr << "Missing required --weights argument\n";
      return 2;
    }

  try
    {
      RngSeedManager::SetSeed (seed);
      auto weights = LoadWeights (weightsPath);
      HrlccTsnMappoSimulation simulation (std::move (weights),
                                          csvPath,
                                          iterations,
                                          seed,
                                          k,
                                          radiusM,
                                          speedMinMps,
                                          speedMaxMps,
                                          hyperperiodMs,
                                          tsnRateMbps,
                                          hrllcBandwidthMbps,
                                          carrierFrequencyGhz,
                                          miniSlotMs,
                                          retransmissionMs,
                                          retransmissionGuardMs,
                                          hopCount,
                                          flow1PeriodMs,
                                          flow1LatencyMs,
                                          flow1Bytes,
                                          flow1Epsilon,
                                          flow2PeriodMs,
                                          flow2LatencyMs,
                                          flow2Bytes,
                                          flow2Epsilon,
                                          totalPowerW,
                                          gammaThresholdDb,
                                          shadowFadingStdDb,
                                          smallScaleFadingRate,
                                          thermalNoiseDensityDbmHz,
                                          pathLossC,
                                          pathLossD0,
                                          pathLossD1,
                                          boundaryTurnMinRad,
                                          boundaryTurnMaxRad,
                                          minimumDistanceM,
                                          lambdaNormalizationOffsetDb,
                                          lambdaNormalizationScaleDb,
                                          lambdaNormalizationMin,
                                          lambdaNormalizationMax,
                                          softmaxMaskLogit,
                                          probabilityFloor,
                                          denominatorFloor,
                                          reliabilityPenaltyWeight,
                                          shadowFadingPerStep,
                                          conservativeRetryFeasibility,
                                          randomGcl,
                                          simpleReward,
                                          ParseCommaSeparatedDoubles (referenceGainsCsv),
                                          deterministic,
                                          runIndex);
      simulation.Run ();
    }
  catch (const std::exception& ex)
    {
      std::cerr << "hrllc_tsn_mappo failed: " << ex.what () << "\n";
      return 1;
    }

  return 0;
}
