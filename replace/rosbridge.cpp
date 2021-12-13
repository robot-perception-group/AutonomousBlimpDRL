/*
 ******************************************************************************
 * @addtogroup UAVOROSBridge UAVO to ROS Bridge Module
 * @{
 *
 * @file       rosbridge.cpp
 * @author     The LibrePilot Project, http://www.librepilot.org Copyright (C) 2016.
 *             Max Planck Institute for intelligent systems, http://www.is.mpg.de Copyright (C) 2016.
 * @brief      Bridges certain UAVObjects to ROS on USB VCP
 *
 * @see        The GNU Public License (GPL) Version 3
 *
 *****************************************************************************/
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * Additional note on redistribution: The copyright and license notices above
 * must be maintained in each individual source file that is a derivative work
 * of this source file; otherwise redistribution is prohibited.
 */

#include "rosbridge.h"
#include "ros/ros.h"
#include "boost/asio.hpp"
#include <boost/lexical_cast.hpp>
#include "boost/thread.hpp"
#include "anonymoussocket.h"
#include "readthread.h"
#include "writethread.h"
#include <string>

namespace librepilot {
rosbridge *globalRosbridge;

/*
        void sigHandler(const boost::system::error_code& error, int signal_number) {
                // shutdown read and write threads too
                exit(-1);
                ros::shutdown();
        }
 */

class rosbridge_priv {
public:
    int argc;
    char * *argv;
    ros::NodeHandle *nodehandle = NULL;
    uint8_t mySequenceNumber;
    anonymoussocket socket;
    boost::asio::io_service io_service;
    boost::posix_time::ptime start;
    boost::mutex serial_Mutex;
    boost::mutex ROSinfo_Mutex;
    std::string nameSpace;
    boost::thread *thread;
    volatile int canary;
    offset3d offset;
    // ...
    void run()
    {
        while (1) {
            this->canary = 0;
            boost::this_thread::sleep(boost::posix_time::seconds(10));
            if (this->canary == 0) {
                fprintf(stderr, "CRITICAL TIMEOUT FAILURE! BAILING OUT!");
                //exit(1); ## YuTang
            }
        }
    }
};

rosbridge::rosbridge(int argc, char * *argv)
{
    globalRosbridge  = this;
    instance = new rosbridge_priv();
    instance->argc   = argc;
    instance->argv   = argv;
    instance->start  = boost::posix_time::microsec_clock::universal_time();
    instance->thread = new boost::thread(boost::bind(&rosbridge_priv::run, instance));
}

rosbridge::~rosbridge()
{
    if (instance->nodehandle) {
        delete instance->nodehandle;
    }
    delete instance;
}

int rosbridge::run(void)
{
    ros::init(instance->argc, instance->argv, "librepilot");

    instance->nodehandle = new ros::NodeHandle();
    // boost::asio::signal_set signals(instance->io_service, SIGINT, SIGTERM);
    // signals.async_wait(sigHandler);

    if (instance->argc < 4) {
        printf("Usage: %s <namespace> <serial_port> <baudrate>\n", instance->argv[0]);
        printf("or\n");
        printf("Usage: %s <namespace> UDP <server> <port>\n", instance->argv[0]);
        return -1;
    }

    instance->nameSpace = std::string(instance->argv[1]);

    if (std::string(instance->argv[2]) == std::string("UDP")) {
        instance->socket.open_udp(std::string(instance->argv[3]), std::string(instance->argv[4]));
    } else {
        // open tty device
        instance->socket.open_serial(std::string(instance->argv[2]), std::string(instance->argv[3]));
    }

    readthread reader(instance->nodehandle, boost::shared_ptr<anonymoussocket>(&instance->socket), this);
    writethread writer(instance->nodehandle, this);
    ros::AsyncSpinner spinner(4);
    spinner.start();
    ros::waitForShutdown();

    // join the other threads
    return 0;
}

boost::posix_time::ptime *rosbridge::getStart(void)
{
    return &instance->start;
}

void rosbridge::setMySequenceNumber(uint8_t value)
{
    instance->mySequenceNumber = value;
}
uint8_t rosbridge::getMySequenceNumber(void)
{
    return instance->mySequenceNumber;
}

int rosbridge::serialWrite(uint8_t *buffer, size_t length)
{
    instance->serial_Mutex.lock();
    int res = instance->socket.write(buffer, length);
    instance->serial_Mutex.unlock();
    instance->canary = 1;


    return res;
}

void rosbridge::rosinfoPrint(const char *bla)
{
    instance->ROSinfo_Mutex.lock();
    ROS_INFO("%s", bla);
    instance->ROSinfo_Mutex.unlock();
}
void rosbridge::setOffset(offset3d &offset)
{
    instance->ROSinfo_Mutex.lock();
    instance->offset = offset;
    instance->ROSinfo_Mutex.unlock();
}
offset3d rosbridge::getOffset()
{
    instance->ROSinfo_Mutex.lock();
    offset3d tmp = instance->offset;
    instance->ROSinfo_Mutex.unlock();
    return tmp;
}

std::string rosbridge::getNameSpace()
{
    return instance->nameSpace;
}
}
