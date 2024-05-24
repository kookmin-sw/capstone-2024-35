require('dotenv').config();
const connectDB = require('./config/db');
const Career = require('./models/Career')
const Employee = require('./models/Employee')
const User = require('./models/User')
const Worksite = require('./models/Worksite')
const mongoose = require('mongoose')
const moment = require('moment')
const { randBirth } = require('./utils/utils')
const utils = require('./utils/utils')

connectDB();

// 테스트용 시드 데이터를 초기화하는 파일
const makeSeeds = async () => {
    const user = await User.findOne({});

    // 근로자 콜렉션 초기화
    await Employee.deleteMany({})
    
    for (i=1; i<=24; i++) {
        const name = utils.getRandName()
        const {address, local} = utils.getRandAddressAndLocal()
        const RRN = utils.getRandBirth()
        const phone = utils.getRandPhone()
        // const isExistName = Employee.find({name:name})
        const newEmp = new Employee({user: user, name: name, sex: '남', local: local, RRN: RRN, phonenumber: phone})

        await newEmp.save()
    }

    // 현장 콜렉션 초기화
    await Worksite.deleteMany({})

    const emps = await Employee.find({})
    for (i=1; i<=10; i++) {
        const {address, local} = utils.getRandAddressAndLocal()
        const sal = 110000 + ((Math.floor(Math.random()*5))*10000)
        const worktype = utils.getRandWorktype()
        const date = new Date()
        date.setDate(date.getDate()+(Math.floor(Math.random()*14)+2))
        const end = new Date(date)
        end.setHours(end.getHours()+(Math.floor(Math.random()*2))+7)
        const nopr = Math.floor(Math.random()*13)+3

        const newWork = new Worksite({user: user, name: '현장'+i, address: address, local: local, salary: sal, worktype: worktype, date: date, end: end, nopr: nopr, worksitenote: '현장 메모'})
        
        // 현장에 랜덤 인부 배정
        if (i <= 10) {
            const ranNopr = Math.floor(Math.random()*nopr)
            for (j=1; j<=(nopr-ranNopr); j++) {
                let index = Math.floor(Math.random()*emps.length)

                // 이미 추가된 근로자는 제외
                let isAlreadyHired = false
                newWork.hired.some(emp => {
                    if (emp._id == emps[index]._id) {
                        isAlreadyHired = true
                        return true
                    }
                });

                // console.log('isAlreadyHired : ', isAlreadyHired)
                // if (isAlreadyHired) {
                //     console.log('hired : ', newWork.hired)
                //     console.log('to push : ', emps[index])
                // }

                if (!isAlreadyHired) {
                    newWork.hired.push(emps[index])
                    newWork.recieved.push({phone: emps[index].phonenumber, message: '가능합니다'})
                    const career = new Career({employee: emps[index], worksite: newWork, pay: sal, review: ''})
                    await career.save()
                }
                
            }
        }
        await newWork.save()
    }
}

makeSeeds().then(() => {
    mongoose.connection.close();
})