require('dotenv').config();
const connectDB = require('./config/db');
const Career = require('./models/Career')
const Employee = require('./models/Employee')
const User = require('./models/User')
const Worksite = require('./models/Worksite')
const mongoose = require('mongoose')

connectDB();

// 테스트용 시드 데이터를 초기화하는 파일
const makeSeeds = async () => {
    const user = await User.findOne({});

    // 근로자 콜렉션 초기화
    await Employee.deleteMany({})
    for (i=1; i<=20; i++) {
        const newEmp = new Employee({user: user, name: '근로자'+i, sex: '남', local: '경기도', RRN: '000101-0000000', phonenumber: '010-1234-5678'})
        await newEmp.save()
    }

    // 현장 콜렉션 초기화
    await Worksite.deleteMany({})
    const date = new Date()
    date.setDate(date.getDate()+7)
    // date.setHours(8)
    const end = new Date(date)
    end.setHours(end.getHours()+8)


    const emps = await Employee.find({})
    for (i=1; i<=10; i++) {
        const newWork = new Worksite({user: user, name: '임시현장'+i, address: '국민대학교 미래관', local: '서울 성북구', salary: '110000', worktype: '이삿짐 운반', date: date, end: end, nopr: 10})
        
        // 1~3번 현장에 랜덤 인부 배정
        if (i <= 3) {
            for (j=1; j<=3; j++) {
                let ran = Math.floor(Math.random()*emps.length)
                newWork.hired.push(emps[ran])
                const career = new Career({employee: emps[ran], worksite: newWork})
                await career.save()
            }
            
        }
        await newWork.save()
    }

}

makeSeeds().then(() => {
    mongoose.connection.close();
})