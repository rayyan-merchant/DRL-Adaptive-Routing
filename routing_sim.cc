/*
 * Project: DRL-Based Adaptive Network Routing
 * File: routing_sim.cc
 * Purpose: ns-3 baseline simulation with Dijkstra routing and FlowMonitor
 * Authors: Muhammad Sabeeh (23K-0002), Rayyan Merchant (23K-0073)
 * Note: This file is used for Phase 1 (baseline). Phase 2 adds ns3-gym hooks.
 */

#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/applications-module.h>
#include <ns3/flow-monitor-module.h>
#include <ns3/ipv4-static-routing-helper.h>
#include <ns3/ipv4-global-routing-helper.h>
#include "routing_env.h"
#include "ns3/opengym-module.h"
#include <sstream>
#include <string>

using namespace ns3;

// Global variable removed - now captured in lambda

void TriggerLinkFailure(Ptr<NetDevice> dev, Ptr<RateErrorModel> failModel) {
    dev->SetAttribute("ReceiveErrorModel", PointerValue(failModel));
}

int main(int argc, char *argv[])
{
    double simTime      = 100.0;   // total simulation time in seconds
    bool   enableRL     = false;   // true when running with DRL agent
    bool   enableFail   = false;   // true for link failure scenario
    double failTime     = 40.0;    // time when R1-D1 link fails
    std::string scenario = "normal";  // normal / congested / failure / mixed
    std::string outFile  = "baseline.xml";  // FlowMonitor output filename
    uint32_t    seed     = 42;     // random seed for reproducibility
    uint32_t    runNum   = 1;      // run number for seed variation

    CommandLine cmd;
    cmd.AddValue("simTime", "Total simulation time in seconds", simTime);
    cmd.AddValue("enableRL", "True when running with DRL agent", enableRL);
    cmd.AddValue("enableFail", "True for link failure scenario", enableFail);
    cmd.AddValue("failTime", "Time when R1-D1 link fails", failTime);
    cmd.AddValue("scenario", "normal / congested / failure / mixed", scenario);
    cmd.AddValue("output", "FlowMonitor output filename", outFile);
    cmd.AddValue("seed", "Random seed for reproducibility", seed);
    cmd.AddValue("runNum", "Run number for seed variation", runNum);
    cmd.Parse(argc, argv);
    std::cout << "Parsed output path: " << outFile << std::endl;

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(runNum);

    // === TOPOLOGY SETUP STARTS HERE ===
    // Nodes: S1=0, S2=1, R1=2, R2=3, R3=4, D1=5
    NodeContainer nodes;
    nodes.Create(6);

    struct LinkDef {
        uint32_t u, v;
        const char* bw;
        const char* delay;
    };

    LinkDef linkDefs[10] = {
        {0, 2, "10Mbps", "2ms"},    // index 0: S1-R1
        {0, 3, "7Mbps",  "4ms"},    // index 1: S1-R2
        {1, 3, "7Mbps",  "4ms"},    // index 2: S2-R2
        {1, 4, "10Mbps", "2ms"},    // index 3: S2-R3
        {2, 3, "5Mbps",  "5ms"},    // index 4: R1-R2 bottleneck
        {3, 4, "5Mbps",  "5ms"},    // index 5: R2-R3 bottleneck
        {2, 5, "8Mbps",  "8ms"},    // index 6: R1-D1 PRIMARY EXIT (fails in failure scenario)
        {3, 5, "6Mbps",  "6ms"},    // index 7: R2-D1
        {4, 5, "4Mbps",  "10ms"},   // index 8: R3-D1
        {2, 4, "3Mbps",  "12ms"}    // index 9: R1-R3 low-cap
    };

    NetDeviceContainer devs[10];
    PointToPointHelper p2p;

    for (int i = 0; i < 10; ++i) {
        p2p.SetDeviceAttribute("DataRate", StringValue(linkDefs[i].bw));
        p2p.SetChannelAttribute("Delay", StringValue(linkDefs[i].delay));
        devs[i] = p2p.Install(nodes.Get(linkDefs[i].u), nodes.Get(linkDefs[i].v));
    }

    // === INTERNET STACK AND IP ADDRESSING ===
    InternetStackHelper inet;
    inet.Install(nodes);

    Ipv4AddressHelper addr;
    for (int i = 0; i < 10; ++i) {
        std::ostringstream subnet;
        subnet << "10." << i << ".1.0";
        addr.SetBase(subnet.str().c_str(), "255.255.255.0");
        addr.Assign(devs[i]);
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // D1 is node index 5, interface 1 is the first assigned interface (interface 0 is loopback).
    Ipv4Address d1_ip = nodes.Get(5)->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();

    // === TRAFFIC GENERATION ===
    uint16_t port = 9;

    ApplicationContainer sinkApps;
    PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    sinkApps.Add(sink.Install(nodes.Get(5)));
    sinkApps.Start(Seconds(0.0));
    sinkApps.Stop(Seconds(simTime));

    std::string s1Rate;
    std::string s2Rate;

    if (scenario == "congested") {
        s1Rate = "9Mbps";
        s2Rate = "8Mbps";
    } else {
        s1Rate = "4Mbps";
        s2Rate = "3Mbps";
    }

    OnOffHelper onoff1("ns3::UdpSocketFactory", InetSocketAddress(d1_ip, port));
    onoff1.SetAttribute("DataRate", StringValue(s1Rate));
    onoff1.SetAttribute("PacketSize", UintegerValue(1024));
    onoff1.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    onoff1.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    ApplicationContainer app1 = onoff1.Install(nodes.Get(0));
    app1.Start(Seconds(1.0));
    app1.Stop(Seconds(simTime - 0.1));

    OnOffHelper onoff2("ns3::UdpSocketFactory", InetSocketAddress(d1_ip, port));
    onoff2.SetAttribute("DataRate", StringValue(s2Rate));
    onoff2.SetAttribute("PacketSize", UintegerValue(1024));
    onoff2.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    onoff2.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    ApplicationContainer app2 = onoff2.Install(nodes.Get(1));
    app2.Start(Seconds(1.0));
    app2.Stop(Seconds(simTime - 0.1));

    // TODO Phase 1 extension: implement mixed scenario using
    // Simulator::Schedule to change DataRate at t=30s and t=60s
    // For now, mixed uses the same rate as normal.

    // === LINK FAILURE SCENARIO ===
    if (enableFail) {
        Ptr<RateErrorModel> failModel = CreateObject<RateErrorModel>();
        failModel->SetAttribute("ErrorRate", DoubleValue(1.0));
        failModel->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));

        // In ns-3.35, Simulator::Schedule doesn't support lambdas directly.
        // We pass the callback parameters explicitly.
        Simulator::Schedule(Seconds(failTime), &TriggerLinkFailure, devs[6].Get(1), failModel);
    }

    FlowMonitorHelper fmHelper;
    Ptr<FlowMonitor> flowMon = fmHelper.InstallAll();

    Ptr<OpenGymInterface> openGym;
    if (enableRL) {
        // ns-3.35 TypeId system requires default constructor, then Setup()
        Ptr<RoutingEnv> routingEnv = CreateObject<RoutingEnv>();
        routingEnv->Setup(
            nodes, devs, d1_ip, flowMon,
            5.0,     // tmon: 5 simulated seconds per step
            simTime, // total simulation time
            3        // kPaths
        );

        openGym = CreateObject<OpenGymInterface>(5555);
        openGym->SetGetActionSpaceCb(MakeCallback(&RoutingEnv::GetActionSpace, routingEnv));
        openGym->SetGetObservationSpaceCb(MakeCallback(&RoutingEnv::GetObservationSpace, routingEnv));
        openGym->SetGetObservationCb(MakeCallback(&RoutingEnv::GetObservation, routingEnv));
        openGym->SetGetRewardCb(MakeCallback(&RoutingEnv::GetReward, routingEnv));
        openGym->SetGetGameOverCb(MakeCallback(&RoutingEnv::GetGameOver, routingEnv));
        openGym->SetGetExtraInfoCb(MakeCallback(&RoutingEnv::GetExtraInfo, routingEnv));
        openGym->SetExecuteActionsCb(MakeCallback(&RoutingEnv::ExecuteActions, routingEnv));

        routingEnv->UpdateCachedMetrics();
        openGym->NotifyCurrentState();
        Simulator::Schedule(Seconds(5.0), &RoutingEnv::ScheduleNextStep, routingEnv);
    }

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    flowMon->SerializeToXmlFile(outFile, true, true);
    // true, true = enableHistograms, enableProbes

    Simulator::Destroy();

    if (enableRL && openGym) {
        openGym->NotifySimulationEnd();
    }

    std::cout << "Simulation complete. Output: " << outFile << std::endl;

    return 0;
}
