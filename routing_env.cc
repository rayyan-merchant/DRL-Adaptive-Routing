#include "routing_env.h"
#include "ns3/log.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv4-routing-table-entry.h"
#include "ns3/ipv4-list-routing.h"
#include "ns3/simulator.h"
#include "ns3/opengym-module.h"
#include <algorithm>
#include <numeric>

namespace {
    uint64_t g_bytesTx[10] = {0};

    void MacTxCallback(uint32_t linkIndex, ns3::Ptr<const ns3::Packet> packet) {
        g_bytesTx[linkIndex] += packet->GetSize();
    }
}

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("RoutingEnv");

TypeId RoutingEnv::GetTypeId() {
    static TypeId tid = TypeId("RoutingEnv")
        .SetParent<OpenGymEnv>()
        .SetGroupName("OpenGym")
        .AddConstructor<RoutingEnv>();
    return tid;
}

RoutingEnv::RoutingEnv()
{
    m_tmon = 5.0;
    m_simTime = 100.0;
    m_kPaths = 3;
    m_currentSD = 0;
    m_stepCount = 0;
    m_gameOver = false;
}

void RoutingEnv::Setup(NodeContainer nodes,
                       NetDeviceContainer devs[],
                       Ipv4Address d1ip,
                       Ptr<FlowMonitor> flowmon,
                       double tmon,
                       double simTime,
                       uint32_t kPaths)
{
    m_nodes = nodes;
    for (int i = 0; i < 10; i++) {
        m_devs[i] = devs[i];
        
        // Hook the trace source for both ends of the link
        m_devs[i].Get(0)->TraceConnectWithoutContext("MacTx", MakeBoundCallback(&MacTxCallback, (uint32_t)i));
        m_devs[i].Get(1)->TraceConnectWithoutContext("MacTx", MakeBoundCallback(&MacTxCallback, (uint32_t)i));
    }
    m_d1ip = d1ip;
    m_flowmon = flowmon;
    m_tmon = tmon;
    m_simTime = simTime;
    m_kPaths = kPaths;

    m_currentSD = 0;
    m_stepCount = 0;
    m_gameOver = false;

    m_paths = {
        { {0,2,5}, {0,3,5}, {0,2,3,5} },
        { {1,4,5}, {1,3,5}, {1,3,2,5} }
    };

    m_cachedMetrics.resize(9, 0.5f);
}

RoutingEnv::~RoutingEnv() {}

Ptr<OpenGymSpace> RoutingEnv::GetObservationSpace() {
    std::vector<uint32_t> shape_vec = {10};
    return CreateObject<OpenGymBoxSpace>(0.0f, 1.0f, shape_vec, "float32");
}

Ptr<OpenGymSpace> RoutingEnv::GetActionSpace() {
    return CreateObject<OpenGymDiscreteSpace>(m_kPaths);
}

void RoutingEnv::UpdateCachedMetrics() {
    m_flowmon->CheckForLostPackets();
    // We don't actually use per-flow stats for the simplified proxy,
    // but we call it so FlowMonitor updates internal counters.
    // auto stats = m_flowmon->GetFlowStats();

    double link_capacity_mbps[10] = {10, 7, 7, 10, 5, 5, 8, 6, 4, 3};
    double delay_ms[10] = {2, 4, 4, 2, 5, 5, 8, 6, 10, 12};

    double link_load[10] = {0};
    double available_bw_mbps[10] = {0};
    double loss_ratio[10] = {0};

    // Calculate actual link load based on transmitted bytes
    for (int i = 0; i < 10; ++i) {
        double mbps = (g_bytesTx[i] * 8.0) / (m_tmon * 1e6);
        link_load[i] = std::min(1.0, mbps / link_capacity_mbps[i]);
        
        // Reset counter for the next interval
        g_bytesTx[i] = 0;

        available_bw_mbps[i] = (1.0 - link_load[i]) * link_capacity_mbps[i];
        loss_ratio[i] = link_load[i] * 0.1;
    }

    struct LinkEndpoints { uint32_t u, v; };
    LinkEndpoints linkDefs[10] = {
        {0,2}, {0,3}, {1,3}, {1,4}, {2,3}, {3,4}, {2,5}, {3,5}, {4,5}, {2,4}
    };

    std::vector<float> raw_metrics(9, 0.0f);

    for (uint32_t p = 0; p < 3; ++p) {
        std::vector<uint32_t>& path = m_paths[m_currentSD][p];
        double bw_path = 1e9;
        double delay_path = 0.0;
        double loss_path = 1.0;

        for (size_t i = 0; i < path.size() - 1; ++i) {
            uint32_t u = path[i];
            uint32_t v = path[i+1];
            int link_idx = -1;

            for (int l = 0; l < 10; ++l) {
                if ((linkDefs[l].u == u && linkDefs[l].v == v) ||
                    (linkDefs[l].u == v && linkDefs[l].v == u)) {
                    link_idx = l;
                    break;
                }
            }

            if (link_idx != -1) {
                bw_path = std::min(bw_path, available_bw_mbps[link_idx]);
                delay_path += delay_ms[link_idx];
                loss_path *= (1.0 - loss_ratio[link_idx]);
            }
        }

        loss_path = 1.0 - loss_path;

        raw_metrics[p*3 + 0] = (float)bw_path;
        raw_metrics[p*3 + 1] = (float)delay_path;
        raw_metrics[p*3 + 2] = (float)loss_path;
    }

    // Min-Max normalize
    for (int c = 0; c < 3; ++c) {
        float min_val = raw_metrics[0*3 + c];
        float max_val = raw_metrics[0*3 + c];
        for (int p = 1; p < 3; ++p) {
            min_val = std::min(min_val, raw_metrics[p*3 + c]);
            max_val = std::max(max_val, raw_metrics[p*3 + c]);
        }

        for (int p = 0; p < 3; ++p) {
            if (max_val - min_val < 1e-6) {
                raw_metrics[p*3 + c] = 0.5f;
            } else {
                raw_metrics[p*3 + c] = (raw_metrics[p*3 + c] - min_val) / (max_val - min_val);
            }
        }
    }

    m_cachedMetrics = raw_metrics;
}

Ptr<OpenGymDataContainer> RoutingEnv::GetObservation() {
    auto box = CreateObject<OpenGymBoxContainer<float> >();

    box->AddValue((float)m_currentSD);
    for (int i = 0; i < 9; ++i) {
        box->AddValue(m_cachedMetrics[i]);
    }
    return box;
}

float RoutingEnv::GetReward() {
    return 0.0f;
}

bool RoutingEnv::GetGameOver() {
    m_gameOver = (Simulator::Now().GetSeconds() >= m_simTime - 0.1);
    return m_gameOver;
}

std::string RoutingEnv::GetExtraInfo() {
    std::ostringstream ss;
    ss << "step=" << m_stepCount << ",sd=" << m_currentSD;
    return ss.str();
}

namespace {
    uint32_t FindInterface(NodeContainer nodes, NetDeviceContainer devs[], uint32_t fromNode, uint32_t toNode) {
        static uint32_t linkU[] = {0,0,1,1,2,3,2,3,4,2};
        static uint32_t linkV[] = {2,3,3,4,3,4,5,5,5,4};

        for (int i = 0; i < 10; ++i) {
            if ((linkU[i] == fromNode && linkV[i] == toNode) ||
                (linkU[i] == toNode && linkV[i] == fromNode)) {

                Ptr<NetDevice> dev = (linkU[i] == fromNode) ? devs[i].Get(0) : devs[i].Get(1);
                return nodes.Get(fromNode)->GetObject<Ipv4>()->GetInterfaceForDevice(dev);
            }
        }
        return 1;
    }

    void RemoveHostRoute(Ptr<Ipv4StaticRouting> sr, Ipv4Address dest) {
        for (uint32_t i = 0; i < sr->GetNRoutes(); ++i) {
            if (sr->GetRoute(i).GetDest() == dest) {
                sr->RemoveRoute(i);
                return;
            }
        }
    }
}

void RoutingEnv::UpdateRouting(uint32_t pathIdx) {
    std::vector<uint32_t>& path = m_paths[m_currentSD][pathIdx];

    for (size_t i = 0; i < path.size() - 1; ++i) {
        uint32_t curNode = path[i];
        uint32_t nextNode = path[i+1];

        Ptr<Ipv4> ipv4 = m_nodes.Get(curNode)->GetObject<Ipv4>();
        Ptr<Ipv4RoutingProtocol> rp = ipv4->GetRoutingProtocol();
        Ptr<Ipv4StaticRouting> staticRouting = 0;
        
        if (Ptr<ns3::Ipv4ListRouting> listRp = DynamicCast<ns3::Ipv4ListRouting>(rp)) {
            int16_t priority;
            for (uint32_t j = 0; j < listRp->GetNRoutingProtocols(); j++) {
                Ptr<Ipv4RoutingProtocol> tempRp = listRp->GetRoutingProtocol(j, priority);
                if (DynamicCast<Ipv4StaticRouting>(tempRp)) {
                    staticRouting = DynamicCast<Ipv4StaticRouting>(tempRp);
                    break;
                }
            }
        } else {
            staticRouting = DynamicCast<Ipv4StaticRouting>(rp);
        }

        if (!staticRouting) {
            NS_LOG_ERROR("Could not find Ipv4StaticRouting protocol!");
            continue;
        }

        uint32_t iface = FindInterface(m_nodes, m_devs, curNode, nextNode);

        RemoveHostRoute(staticRouting, m_d1ip);
        staticRouting->AddHostRouteTo(m_d1ip, iface, 1);

        break; // Only set next-hop from the source node
    }
}

bool RoutingEnv::ExecuteActions(Ptr<OpenGymDataContainer> action) {
    Ptr<OpenGymDiscreteContainer> disc = DynamicCast<OpenGymDiscreteContainer>(action);
    uint32_t pathIdx = disc->GetValue();

    if (pathIdx >= m_kPaths) pathIdx = 0;

    UpdateRouting(pathIdx);

    m_currentSD = (m_currentSD + 1) % 2;
    m_stepCount++;
    return true;
}

void RoutingEnv::ScheduleNextStep() {
    UpdateCachedMetrics();
    Notify();
    Simulator::Schedule(Seconds(m_tmon), &RoutingEnv::ScheduleNextStep, this);
}

} // namespace ns3