#ifndef ROUTING_ENV_H
#define ROUTING_ENV_H

#include "ns3/opengym-module.h"
#include "ns3/node-container.h"
#include "ns3/net-device-container.h"
#include "ns3/ipv4-address.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/flow-monitor.h"
#include <vector>
#include <map>
#include <string>

namespace ns3 {

class RoutingEnv : public OpenGymEnv {
public:
    static TypeId GetTypeId();

    RoutingEnv();  // default constructor required by ns3 TypeId system

    void Setup(NodeContainer nodes,
               NetDeviceContainer devs[],
               Ipv4Address d1ip,
               Ptr<FlowMonitor> flowmon,
               double tmon,
               double simTime,
               uint32_t kPaths);

    virtual ~RoutingEnv();

    Ptr<OpenGymSpace> GetObservationSpace() override;
    Ptr<OpenGymSpace> GetActionSpace() override;
    Ptr<OpenGymDataContainer> GetObservation() override;
    float GetReward() override;
    bool GetGameOver() override;
    std::string GetExtraInfo() override;
    bool ExecuteActions(Ptr<OpenGymDataContainer> action) override;

    void ScheduleNextStep();
    void UpdateRouting(uint32_t pathIdx);
    void UpdateCachedMetrics();

private:
    NodeContainer m_nodes;
    NetDeviceContainer m_devs[10];
    Ipv4Address m_d1ip;
    Ptr<FlowMonitor> m_flowmon;
    double m_tmon;
    double m_simTime;
    uint32_t m_kPaths;
    uint32_t m_currentSD;
    uint32_t m_stepCount;
    bool m_gameOver;

    std::vector<float> m_cachedMetrics;

    // Outer: SD pair index (0 or 1)
    // Middle: path index (0, 1, 2)
    // Inner: sequence of node indices on the path
    std::vector<std::vector<std::vector<uint32_t>>> m_paths;
};

} // namespace ns3

#endif // ROUTING_ENV_H