exports.getRandBirth = () => {
    const year = 99 - Math.floor(Math.random()*25)
    const month = Math.floor(Math.random()*12) + 1
    const date = Math.floor(Math.random()*28) + 1
    let result = `${year}`
    if (month < 10) {
        result = result + '0' + month
    } else {
        result = result + month
    }

    if (date < 10) {
        result = result + 0 + date
    } else {
        result = result + date
    }
    result = result + '-1'
    
    return result
}

exports.getRandPhone = () => {
    const num1 = Math.floor(Math.random()*9000) + 1000
    const num2 = Math.floor(Math.random()*9000) + 1000
    const result = `010-${num1}-${num2}`
    return result
}

exports.getRandAddressAndLocal = () => {
    const address = [
        '서울특별시 강남구','서울특별시 강동구','서울특별시 강북구','서울특별시 강서구','서울특별시 관악구','서울특별시 광진구','서울특별시 구로구','서울특별시 금천구','서울특별시 노원구','서울특별시 도봉구','서울특별시 동대문구','서울특별시 동작구','서울특별시 마포구','서울특별시 서대문구','서울특별시 서초구','서울특별시 성동구','서울특별시 성북구','서울특별시 송파구','서울특별시 양천구','서울특별시 영등포구','서울특별시 용산구','서울특별시 은평구','서울특별시 종로구','서울특별시 중구','서울특별시 중랑구',
        '경기도 가평군','경기도 고양시','경기도 고양시 덕양구','경기도 고양시 일산서구','경기도 고양시 일산동구','경기도 과천시','경기도 광명시','경기도 광주시','경기도 구리시','경기도 군포시','경기도 김포시','경기도 남양주시','경기도 동두천시','경기도 부천시','경기도 부천시 오정구','경기도 부천시 원미구','경기도 부천시 소사구','경기도 성남시','경기도 성남시 중원구','경기도 성남시 수정구','경기도 성남시 분당구','경기도 수원시','경기도 수원시 장안구','경기도 수원시 권선구','경기도 수원시 팔달구','경기도 수원시 영통구','경기도 시흥시','경기도 안산시','경기도 안산시 단원구','경기도 안산시 상록구','경기도 안성시','경기도 안양시','경기도 안양시 만안구','경기도 안양시 동안구','경기도 양주시','경기도 양평군','경기도 여주시','경기도 연천군','경기도 오산시','경기도 용인시','경기도 용인시 처인구','경기도 용인시 기흥구','경기도 용인시 수지구','경기도 의왕시','경기도 의정부시','경기도 이천시','경기도 파주시','경기도 평택시','경기도 포천시','경기도 하남시','경기도 화성시',
        '인천광역시','인천광역시 계양구','인천광역시 남동구','인천광역시 동구','인천광역시 미추홀구','인천광역시 부평구','인천광역시 서구','인천광역시 연수구','인천광역시 중구',

    ]
    const locals = [
        // 서울 구는 서울 생략
        '강남구','강동구','강북구','강서구','관악구','광진구','구로구','금천구','노원구','도봉구','동대문구','동작구','마포구','서대문구','서초구','성동구','성북구','송파구','양천구','영등포구','용산구','은평구','종로구','중구','중랑구',
        // 경기도 일반 시는 경기도 생략
        '가평군','고양시','고양시 덕양구','고양시 일산서구','고양시 일산동구','과천시','광명시','광주시','구리시','군포시','김포시','남양주시','동두천시','부천시','부천시 오정구','부천시 원미구','부천시 소사구','성남시','성남시 중원구','성남시 수정구','성남시 분당구','수원시','수원시 장안구','수원시 권선구','수원시 팔달구','수원시 영통구','시흥시','안산시','안산시 단원구','안산시 상록구','안성시','안양시','안양시 만안구','안양시 동안구','양주시','양평군','여주시','연천군','오산시','용인시','용인시 처인구','용인시 기흥구','용인시 수지구','의왕시','의정부시','이천시','파주시','평택시','포천시','하남시','화성시',
        '인천','인천 계양구','인천 남동구','인천 동구','인천 미추홀구','인천 부평구','인천 서구','인천 연수구','인천 중구',
    ]
    const index = Math.floor(Math.random()*locals.length)
    return {address: address[index], local: locals[index]}
}

exports.getRandName = () => {
    // 이름은 랜덤 생성기로 30개 생성
    const names = ['봉동근', '백지호', '성민웅', '임지석', '황수정', '황다영', '풍은식', '백범석', '허효민', '전혜지', '김성빈', '김수민', '예재민', '전경민', '노영우', '이혜자', '강규희', '안윤희', '정명우', '하상현', '풍인정', '하병옥', '정희원', '조동하', '풍민정', '신정우', '장재선', '문창수', '안재근', '손유원', '고광석', '유정환', '양진미', '유혜준', '노경숙', '류우연', '서광희', '허선화', '배성용', '복성준', '최만희', '백진선', '김보연', '안대현', '정상아', '봉동영', '권명환', '황재윤', '이창호', '서미자', '임태영', '하해성', '탁민혁', '풍영현', '오영철', '손지선', '신소훈', '문준영', '복지현', '안종남']
    return names[Math.floor(Math.random()*names.length)]
}

exports.getRandWorktype = () => {
    // 이름은 랜덤 생성기로 30개 생성
    const worktypes = ['운반', '가구 설치', '가정 이사', '사무실 이전', '내부 이동', '사무 집기', '폐기', '철거']
    return worktypes[Math.floor(Math.random()*worktypes.length)]
}

// for (i=0; i<10; i = i+1) {
//     console.log(randPhone())
// }
