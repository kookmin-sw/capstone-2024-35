const mongoose = require('mongoose');


const Schema = mongoose.Schema;
const WorksiteSchema = new Schema({
  user: {
    type: Schema.ObjectId,
    ref: 'User',
    required: true,
  },
  // 현장명
  name: {
    type: String,
    required: true,
  },
  // 현장 상세 주소
  address: {
    type: String,
    required: true,
  },
  // 현장 소재지
  local: {
    type: String,
    required: true,
  },
  // 급여
  salary: {
    type: Number,
    required: true,
  },
  // 업종
  worktype: {
    type: String,
    required: true,
  },
  // 날짜
  date: {
    type: Date,
    required: true,
  },
  // 끝나는 시간
  end: {
    type: Date,
    required: true,
  },
  //필요한 인원 (number of people required)
  nopr: {
    type: Number,
    required: true,
  },
  // 채용 희망 인원 (문자 발송 전)
  wanted: [
    {
      type: Schema.ObjectId,
      ref: 'Employee',
    }
  ],
  // 채용 희망 인원 (문자 발송 후, 수신 전)
  sent: [
    {
      type: Schema.ObjectId,
      ref: 'Employee',
    }
  ],
  sendMessage: [
    {
      phone: {
        type: String,
        required: true,
      },
      message: {
        type: String,
        required: true,
      },
      date: {
        type: Date,
        required: true,
        default: Date.now()
      }
    }
  ],
  // 받은 문자들을 저장
  recieved: [
    {
      phone: {
        type: String,
        required: true,
      },
      message: {
        type: String,
        required: true,
      },
      date: {
        type: Date,
        required: true,
        default: Date.now()
      }
    }
  ],
  // 채용 거절한 인원 (문자 수신 후)
  denied: [
    {
      type: Schema.ObjectId,
      ref: 'Employee',
    }
  ],
  // 채용 확정(수락)한 인원 (문자 수신 후))
  hired: [
    {
      type: Schema.ObjectId,
      ref: 'Employee',
    }
  ],
  //현장 내용 예. 5시간 작업 예상, 경력자 필수 등
  worksitenote: {
    type: String,
    required: false,
  },
  // 최초 작성일
  createdAt: {
    type: Date,
    default: Date.now()
  },
  // 최근 수정일
  updatedAt: {
    type: Date,
    default: Date.now()
  }
});

module.exports = mongoose.model('Worksite', WorksiteSchema);